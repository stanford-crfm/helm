import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def serve_request(raw_request):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").half().to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    runtime = time.time() - start_time
    print(f"Done loading model and tokenizer! Took {runtime:.2f} seconds...")

    encoded_input = tokenizer(raw_request["prompt"], return_tensors="pt").to(device)
    output = model.generate(
        **encoded_input,
        temperature=raw_request["temperature"],
        max_length=raw_request["max_tokens"],
        num_return_sequences=raw_request["num_completions"],
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = output.sequences
    all_logprobs = []
    logprobs_of_chosen_tokens = []
    for i in range(len(sequences[0]) - len(encoded_input.input_ids[0])):
        logprobs = torch.log(torch.nn.functional.softmax(output.scores[i][0]))
        all_logprobs.append(logprobs)
        # Get log probability of chosen token.
        j = i + len(encoded_input.input_ids[0])
        logprobs_of_chosen_tokens.append(logprobs[sequences[0][j]].item())

    # Remove prompt from the start of each sequence if echo_prompt is False.
    if not raw_request["echo_prompt"]:
        sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
    all_tokens = [tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
    all_decoded_text = tokenizer.batch_decode(sequences)

    completions = []
    for (decoded_text, tokens) in zip(all_decoded_text, all_tokens):
        completions.append({"text": decoded_text, "tokens": tokens, "logprobs": logprobs_of_chosen_tokens})

    return {"completions": completions}
