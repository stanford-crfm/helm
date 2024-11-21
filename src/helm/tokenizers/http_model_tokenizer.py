import os
from dataclasses import asdict
from typing import Any, Dict, Optional

from helm.common.cache import Cache, CacheConfig
from helm.common.request import wrap_request_time
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from helm.tokenizers.tokenizer import Tokenizer

import requests


class HTTPModelTokenizer(Tokenizer):
    # Does not inherit from CachingTokenizer because the parameter do_cache
    # can be set to True or False.

    def __init__(
        self,
        cache_config: CacheConfig,
        base_url: str = "http://localhost:8080",
        do_cache: bool = False,
    ):
        self.cache: Optional[Cache] = Cache(cache_config) if do_cache else None
        self.base_url = (
            base_url if not os.environ.get("HELM_HTTP_MODEL_BASE_URL") else os.environ["HELM_HTTP_MODEL_BASE_URL"]
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        # TODO: Does not support encoding
        cache_key = asdict(request)
        raw_request = {
            "text": request.text,
            "truncation": request.truncation,
            "max_length": request.max_length,
        }

        try:

            def do_it() -> Dict[str, Any]:
                url = f"{self.base_url}/tokenize"
                response = requests.post(url, json=raw_request)
                response.raise_for_status()
                response_data = response.json()
                return response_data

            if self.cache:
                result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            else:
                result, cached = do_it(), False
        except Exception as e:
            error: str = f"Local Model error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        cache_key = asdict(request)

        try:

            def do_it() -> Dict[str, Any]:
                url = f"{self.base_url}/decode"
                response = requests.post(url, json={"tokens": request.tokens})
                response.raise_for_status()
                response_data = response.json()
                return response_data

            if self.cache:
                result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            else:
                result, cached = do_it(), False
        except Exception as e:
            error: str = f"Local Model error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
