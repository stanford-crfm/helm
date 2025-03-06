# flake8: noqa
# type: ignore
# fmt: off

import json
import random
import re
from typing import Any, List

import numpy as np
from tqdm import tqdm


# The following code is copied verbatim from:
# https://github.com/NVIDIA/RULER/blob/860f2bd5c0430569f5941176f9f97f95e770b3da/scripts/data/synthetic/qa.py
# under the following license:
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


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


# The following code has been modified from the original source from:
# https://github.com/NVIDIA/RULER/blob/860f2bd5c0430569f5941176f9f97f95e770b3da/scripts/data/synthetic/qa.py
# under the same Apache 2.0 license included above.


def _text_to_tokens(text: str) -> List[int]:
    return re.split(r"\s+", text.strip())


def generate_samples(dataset: str, dataset_path: str, template: str, random_seed: int, pre_samples: int, num_samples: int, tokens_to_generate: int, max_seq_length: int, incremental: int = 10, remove_newline_tab: bool = False): 
    random.seed(random_seed)
    np.random.seed(random_seed)

    if dataset == 'squad':
        qas, docs = read_squad(dataset_path)
    elif dataset == 'hotpotqa':
        qas, docs = read_hotpotqa(dataset_path)
    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    write_jsons = []
    tokens_to_generate = tokens_to_generate
    
    # Find the perfect num_docs
    num_docs = incremental
    
    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length :  
        input_text, answer = generate_input_output(0, num_docs, template=template, random_seed=random_seed, qas=qas, docs=docs)
        # Calculate the number of tokens in the example
        total_tokens = len(_text_to_tokens(input_text + f' {answer}'))
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
                length = len(_text_to_tokens(input_text)) + tokens_to_generate
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
