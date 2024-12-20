import os
from dataclasses import asdict
from typing import Any, Dict

from helm.common.cache import CacheConfig
from helm.common.request import (
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)
from helm.clients.client import CachingClient

import requests


class HTTPModelClient(CachingClient):
    """Implements a simple client for a model being served over HTTP."""

    def __init__(
        self,
        cache_config: CacheConfig,
        base_url: str = "http://localhost:8080",
        timeout: int = 3000,
        do_cache: bool = False,
    ):
        super().__init__(cache_config=cache_config)
        self.base_url = (
            base_url if not os.environ.get("HELM_HTTP_MODEL_BASE_URL") else os.environ["HELM_HTTP_MODEL_BASE_URL"]
        )
        self.timeout = timeout
        self.do_cache = do_cache

    def make_request(self, request: Request) -> RequestResult:
        cache_key = asdict(request)
        # This needs to match whatever we define in pedantic
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = {
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        try:

            def do_it() -> Dict[str, Any]:
                url = f"{self.base_url}/process"
                response = requests.post(url, json=raw_request, timeout=self.timeout)
                response.raise_for_status()
                response_data = response.json()
                return response_data

            if self.do_cache:
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            else:
                response, cached = do_it(), False

            tokens = [Token(text=token["text"], logprob=token["logprob"]) for token in response["tokens"]]
            completions = [GeneratedOutput(text=response["text"], logprob=response["logprob"], tokens=tokens)]

            return RequestResult(
                success=True,
                cached=cached,
                error=None,
                completions=completions,
                embedding=[],
                request_time=response["request_time"],
            )
        except requests.exceptions.RequestException as e:
            error: str = f"Request error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])
