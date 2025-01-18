import os
import requests

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Optional

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

class StanfordHealthCareHTTPModelClient(CachingClient, ABC):
    def __init__(
        self,
        cache_config: CacheConfig,
        deployment: str,
        base_url: str = "http://localhost:8080",
        do_cache: bool = False,
        timeout: int = 3000,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        super().__init__(cache_config=cache_config)
        self.base_url = base_url if not os.environ.get("HELM_HTTP_MODEL_BASE_URL") else os.environ["HELM_HTTP_MODEL_BASE_URL"]
        self.timeout = timeout
        self.do_cache = do_cache
        self.deployment = deployment
        self.model = model
        self.default_headers = {"Ocp-Apim-Subscription-key": api_key}

    def make_request(self, request: Request) -> RequestResult:
        cache_key = asdict(request)
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = self.get_request(request)

        try:
            def do_it() -> Dict[str, Any]:
                url = f"{self.base_url}/{self.deployment}"
                response = requests.post(url, json=raw_request, headers=self.default_headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()

            if self.do_cache:
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            else:
                response, cached = wrap_request_time(do_it)(), False

            completions = self.parse_response(response)

            return RequestResult(
                success=True,
                cached=cached,
                error=None,
                completions=completions,
                embedding=[],
                request_time=response["request_time"],
            )
        except requests.exceptions.RequestException as e:
            return RequestResult(success=False, cached=False, error=f"Request error: {e}", completions=[], embedding=[])

    @abstractmethod
    def get_request(self, request: Request) -> Dict[str, Any]:
        pass

    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> List[GeneratedOutput]:
        pass
