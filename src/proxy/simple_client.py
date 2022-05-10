from typing import List, Dict

from common.cache import Cache

from .client import Client, wrap_request_time
from common.request import Request, RequestResult, Sequence, Token
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult, TokenizationToken


class SimpleClient(Client):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def __init__(self, cache_path: str):
        self.cache = Cache(cache_path)

    @staticmethod
    def tokenize_by_space(text: str) -> List[str]:
        """Simply tokenizes by a single white space."""
        return text.split(" ")

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }

        if request.model_engine == "model1":

            def do_it():
                return self.invoke_model1(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            completions = [
                Sequence(
                    text=text,
                    logprob=logprob,
                    tokens=[Token(text=text, logprob=logprob, top_logprobs=response["completions"])],
                )
                for text, logprob in response["completions"].items()
            ]
        else:
            raise ValueError(f"Invalid model: {request.model}")

        return RequestResult(success=True, cached=False, request_time=0, completions=completions)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        if request.model == "simple/model1":
            raw_tokens: List[str] = SimpleClient.tokenize_by_space(request.text)
            return TokenizationRequestResult(
                cached=False, tokens=[TokenizationToken(text) for text in raw_tokens], text=request.text
            )
        else:
            raise ValueError("Unknown model")

    def invoke_model1(self, raw_request: Dict) -> Dict:
        """
        Example: 7 2 4 6
        Completions (num_completions = 3):
        - 6
        - 4
        - 2
        """
        prompt_tokens: List[str] = SimpleClient.tokenize_by_space(raw_request["prompt"])
        choices = reversed(prompt_tokens[-raw_request["n"] :])
        response = {"completions": dict((text, -i) for i, text in enumerate(choices))}
        return response
