import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def serve_request(raw_request):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    encoded_input = tokenizer(raw_request['prompt'], return_tensors='pt').to(device)
    output = model.generate(
        **encoded_input,
        temperature=raw_request['temperature'],
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True
    )
    gen_tokens = output.sequences
    all_logprobs = []
    for i in range(len(gen_tokens[0]) - len(encoded_input.input_ids[0])):
        logprobs = torch.log(torch.nn.functional.softmax(output.scores[i][0]))
        all_logprobs.append(logprobs)
        # Log probability of selected token is given by:
        # >> j = i + len(encoded_input.input_ids[0])
        # >> print(i, logprobs[gen_tokens[0][j]])
    gen_tokens = [tokenizer.convert_ids_to_tokens(gen_token) for gen_token in gen_tokens]
    return gen_tokens


if __name__ == '__main__':
    raw_request = {
        "prompt": "I am a computer scientist.",
        "temperature": 0.9,
    }
    print(serve_request(raw_request))
