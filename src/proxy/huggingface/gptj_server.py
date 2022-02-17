from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


def serve_request(raw_request):
    encoded_input = tokenizer(raw_request['prompt'], return_tensors='pt')
    output = model.generate(
        **encoded_input,
        temperature=raw_request['temperature'],
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True
    )
    gen_tokens = output.sequences
    gen_tokens = [tokenizer.convert_ids_to_tokens(gen_token) for gen_token in gen_tokens]
    return gen_tokens


if __name__ == '__main__':
    raw_request = {
        "prompt": "I am a computer scientist.",
        "temperature": 0.9,
    }
    print(serve_request(raw_request))
