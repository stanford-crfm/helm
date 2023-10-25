from typing import List, Dict

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from helm.proxy.tokenizers.simple_tokenizer import SimpleTokenizer
from helm.proxy.tokenizers.tokenizer import Tokenizer
from .client import CachingClient


class SimpleClient(CachingClient):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def __init__(self, tokenizer: Tokenizer, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config, tokenizer=tokenizer)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }

        if request.model_engine == "model1":

            def do_it():
                return self.invoke_model1(raw_request)

            cache_key = CachingClient.make_cache_key(raw_request, request)
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

        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def invoke_model1(self, raw_request: Dict) -> Dict:
        """
        Example: 7 2 4 6
        Completions (num_completions = 3):
        - 6
        - 4
        - 2
        """
        prompt_tokens: List[str] = SimpleTokenizer.tokenize_by_space(raw_request["prompt"])
        choices = reversed(prompt_tokens[-raw_request["n"] :])
        response = {"completions": dict((text, -i) for i, text in enumerate(choices))}
        return response
