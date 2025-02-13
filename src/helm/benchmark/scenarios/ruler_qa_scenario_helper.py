from tqdm import tqdm
from typing import Any, List
import json
import numpy as np
import random

# Load Tokenizer

class _Tokenizer:
    def text_to_tokens(self, text: str) -> List[int]:
        pass

    def tokens_to_text(self, tokens: List[int]) -> str:
        pass

class _OpenAITokenizer(_Tokenizer):
    """
    Tokenizer from tiktoken
    """
    def __init__(self, model_path="cl100k_base") -> None:
        import tiktoken
        self.tokenizer = tiktoken.get_encoding(model_path)

    def text_to_tokens(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens)
        return text

# Read SQuAD QA dataset
def read_squad(file):
    with open(file) as f:
        data = json.load(f)
        
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
                        
    return total_qas, total_docs

# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
        
    return total_qas, total_docs


DOCUMENT_PROMPT = "Document {i}:\n{document}"

def generate_input_output(index, num_docs, template: str, random_seed: int, qas: Any, docs: Any):
    curr_q = qas[index]['query']
    curr_a = qas[index]['outputs']
    curr_docs = qas[index]['context']
    curr_more = qas[index].get('more_context', [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(docs) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
    
        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs
        
    random.Random(random_seed).shuffle(all_docs)
    
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    input_text = template.format(
        context=context, 
        query=curr_q
    )
    return input_text, curr_a


def generate_samples(tokenizer: _Tokenizer, template: str, random_seed: int, qas: Any, docs: Any, pre_samples: int, num_samples: int, tokens_to_generate: int, max_seq_length: int, incremental: int = 10, remove_newline_tab: bool = False): 
    
    write_jsons = []
    tokens_to_generate = tokens_to_generate
    
    # Find the perfect num_docs
    num_docs = incremental
    
    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length :  
        input_text, answer = generate_input_output(0, num_docs, template=template, random_seed=random_seed, qas=qas, docs=docs)
        # Calculate the number of tokens in the example
        total_tokens = len(tokenizer.text_to_tokens(input_text + f' {answer}'))
        print(f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break
            
        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break
    print('Number of documents:', num_docs)
    
    # Generate samples
    for index in tqdm(range(num_samples)):
        used_docs = num_docs
        while(True):
            try:
                input_text, answer = generate_input_output(index + pre_samples, used_docs, template=template, random_seed=random_seed, qas=qas, docs=docs)
                length = len(tokenizer.text_to_tokens(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_docs > incremental:
                    used_docs -= incremental
        
        if remove_newline_tab:
            input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
        
        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length
        }
        write_jsons.append(formatted_output)

    return write_jsons


def generate_samples_for_helm(dataset_path, tokenizer, max_seq_length, tokens_to_generate, num_samples, random_seed, dataset, pre_samples, template):
    
    random.seed(random_seed)
    np.random.seed(random_seed)

    if dataset == 'squad':
        qas, docs = read_squad(dataset_path)
    elif dataset == 'hotpotqa':
        qas, docs = read_hotpotqa(dataset_path)
    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    write_jsons = generate_samples(
        tokenizer=_OpenAITokenizer(),
        template=template,
        random_seed=random_seed,
        qas=qas,
        docs=docs,
        pre_samples=pre_samples,
        num_samples=num_samples,
        tokens_to_generate=tokens_to_generate,
        max_seq_length=max_seq_length, 
    )

    return write_jsons

# def main():
#     template = """Answer the question based on the given documents. Only give me the answer and do not output any other words.

# The following are given documents.

# {context}

# Answer the question based on the given documents. Only give me the answer and do not output any other words.

# Question: {query} Answer:"""
#     generate_samples_for_scenario(tokenizer="cl100k_base", max_seq_length=131072, tokens_to_generate=32, num_samples=500, random_seed=42, dataset="squad", pre_samples=0, template=template)

