import itertools
from typing import List, TypedDict
from typing import Dict, Any

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from helm.clients.client import CachingClient


class SimpleClientRequest(TypedDict):
    engine: str
    prompt: str
    num_completions: int


class SimpleClient(CachingClient):
    """Simple client for tutorials and for debugging."""

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)

    def make_request(self, request: Request) -> RequestResult:
        raw_request: SimpleClientRequest = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "num_completions": request.num_completions,
        }

        def do_it() -> Dict[str, Any]:
            return self.invoke_model(raw_request)

        cache_key = CachingClient.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        logprob = 0
        completions = [
            Sequence(
                text=text,
                logprob=logprob,
                tokens=[Token(text=text, logprob=logprob)],
            )
            for text in response["completions"]
        ]

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def invoke_model(self, raw_request: SimpleClientRequest) -> Dict[str, Any]:
        """
        Example:
        Prompt: 7 2 4 6
        Completions (num_completions = 3):
        - 6
        - 4
        - 2
        """
<<<<<<< HEAD
        prompt_words: List[str] = raw_request["prompt"].split()
        completions = list(itertools.islice(itertools.cycle(reversed(prompt_words)), raw_request["num_completions"]))
        return {"completions": completions}
=======
        prompt_tokens: List[str] = raw_request["prompt"].split(" ")
        choices = reversed(prompt_tokens[-raw_request["n"] :])
        response = {"completions": dict((text, -i) for i, text in enumerate(choices))}
        return response

    def invoke_model_tutorial(self, raw_request: Dict) -> Dict:
        """Always returns: 'The model is generating some text. Hooray, the tutorial works! (Completion {i})'.
        This supports multiple completions.
        """
        response = {
            "completions": dict(
                (f"The model is generating some text. Hooray, the tutorial works! (Completion {i})", -i)
                for i in range(raw_request["n"])
            )
        }
        return response
>>>>>>> Fix SimpleTokenizer
