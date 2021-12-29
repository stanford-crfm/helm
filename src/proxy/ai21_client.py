import requests

from common.cache import Cache
from common.request import Request, RequestResult, Sequence
from .client import Client


# https://studio.ai21.com/docs/api/
class AI21Client(Client):
    def __init__(self, api_key: str, cache_path: str):
        self.api_key = api_key
        self.cache = Cache(cache_path)

    def make_request(self, request: Request) -> RequestResult:
        model = request.model
        if model not in ["ai21/j1-large", "ai21/j1-jumbo"]:
            raise Exception("Invalid model")
        raw_request = {
            "prompt": request.prompt,
            "temperature": request.temperature,
            "numResults": request.num_completions,
            "topKReturn": request.top_k_per_token,
            "maxTokens": request.max_tokens,
            "stopSequences": request.stop_sequences,
        }

        def do_it():
            return requests.post(
                f"https://api.ai21.com/studio/v1/{request.model_engine()}/complete",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=raw_request,
            ).json()

        response, cached = self.cache.get(raw_request, self.wrap_request_time(do_it))

        if "completions" not in response:
            return RequestResult(success=False, cached=False, error=response["detail"], completions=[])

        completions = []
        for completion in response["completions"]:
            completions.append(Sequence(text=completion["data"]["text"]))
        return RequestResult(
            success=True, cached=cached, request_time=response["request_time"], completions=completions
        )
