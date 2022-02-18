from huggingface import gptj_server

from common.cache import Cache
from common.request import Request, RequestResult, Sequence, Token
from client import Client, wrap_request_time


class HuggingFaceClient(Client):
    def __init__(self, server_location: str, cache_path: str):
        self.server_location = server_location
        self.cache = Cache(cache_path)

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

        try:

            def do_it():
                return gptj_server.serve_request(raw_request)

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
    client = HuggingFaceClient(server_location="localhost", cache_path="huggingface_cache")
    print(client.make_request(Request(prompt="I am a computer scientist.", num_completions=2)))
