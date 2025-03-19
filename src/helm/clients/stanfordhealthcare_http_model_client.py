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
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)
from helm.clients.client import CachingClient


class StanfordHealthCareHTTPModelClient(CachingClient, ABC):
    """
    Client for accessing Stanford Health Care models via HTTP requests.

    Configure by setting the following in prod_env/credentials.conf:

    ```
    stanfordhealthcareEndpoint: https://your-domain-name/
    stanfordhealthcareApiKey: your-private-key
    ```
    """

    CREDENTIAL_HEADER_NAME = "Ocp-Apim-Subscription-Key"

    def __init__(
        self,
        cache_config: CacheConfig,
        deployment: str,
        endpoint: str = "http://localhost:8080",
        do_cache: bool = False,
        timeout: int = 3000,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        assert api_key, "API key must be provided"
        self.endpoint = endpoint
        self.timeout = timeout
        self.do_cache = do_cache
        self.deployment = deployment
        self.model = model
        self.default_headers = {StanfordHealthCareHTTPModelClient.CREDENTIAL_HEADER_NAME: api_key}

    def make_request(self, request: Request) -> RequestResult:
        cache_key = asdict(request)
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = self.get_request(request)

        try:

            def do_it() -> Dict[str, Any]:
                url = f"{self.endpoint}/{self.deployment}"
                response = requests.post(url, json=raw_request, headers=self.default_headers, timeout=self.timeout)
                response.raise_for_status()
                response_json = response.json()
                if type(response_json) == list:
                    response_json = {"content": response_json}
                return response_json

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
