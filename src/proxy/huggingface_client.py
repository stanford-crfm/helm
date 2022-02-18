from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from client import Client, wrap_request_time


import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


class HuggingFaceServer:
    def __init__(self, model_name):
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

        start_time: float = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        runtime: float = time.time() - start_time
        print(f"Done loading model and tokenizer! Took {runtime:.2f} seconds...")

    def serve_request(self, raw_request):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt").to(self.device)
        output = self.model.generate(
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
        all_tokens = [self.tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for (decoded_text, tokens) in zip(all_decoded_text, all_tokens):
            completions.append({"text": decoded_text, "tokens": tokens, "logprobs": logprobs_of_chosen_tokens})

        return {"completions": completions}


class HuggingFaceClient(Client):
    def __init__(self, cache_path: str):
        self.cache = Cache(cache_path)
        self.model_stubs: Dict[str, HuggingFaceServer] = {}

    def get_model_stub(self, model_engine):
        if model_engine not in self.model_stubs:
            if model_engine == "gptj_6b":
                self.model_stubs[model_engine] = HuggingFaceServer("EleutherAI/gpt-j-6B")
            elif model_engine == "gpt2":
                self.model_stubs[model_engine] = HuggingFaceServer("gpt2")
            else:
                raise Exception("Unknown model!")
        return self.model_stubs[model_engine]

    def make_request(self, request: Request) -> RequestResult:
        print(request)
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "num_completions": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "echo_prompt": request.echo_prompt,
        }
        model_stub = self.get_model_stub(request.model_engine)

        try:

            def do_it():
                return model_stub.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            for text, logprob in zip(raw_completion["tokens"], raw_completion["logprobs"]):
                tokens.append(Token(text=text, logprob=logprob, top_logprobs={}))
                sequence_logprob += logprob
            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completions.append(completion)
        return RequestResult(
            success=True, cached=cached, request_time=response["request_time"], completions=completions
        )


if __name__ == "__main__":
    client = HuggingFaceClient(cache_path="huggingface_cache")
    print(
        client.make_request(
            Request(model="huggingface/gptj_6b", prompt="I am a computer scientist.", num_completions=2)
        )
    )
    print(
        client.make_request(
            Request(model="huggingface/gpt2", prompt="My name is Joe.", num_completions=1, max_tokens=10)
        )
    )
