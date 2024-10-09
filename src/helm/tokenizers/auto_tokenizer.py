from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from retrying import Attempt, RetryError

from helm.benchmark.tokenizer_config_registry import get_tokenizer_config
from helm.common.credentials_utils import provide_api_key
from helm.common.cache_backend_config import CacheBackendConfig, CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import create_object, inject_object_spec_args
from helm.proxy.retry import retry_tokenizer_request
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.tokenizers.tokenizer import Tokenizer


class AutoTokenizer(Tokenizer):
    """Automatically dispatch to the proper `Tokenizer` based on the tokenizer name."""

    def __init__(self, credentials: Mapping[str, Any], cache_backend_config: CacheBackendConfig):
        self.credentials = credentials
        self.cache_backend_config = cache_backend_config
        self.tokenizers: Dict[str, Tokenizer] = {}
        hlog(f"AutoTokenizer: cache_backend_config = {cache_backend_config}")

    def _get_tokenizer(self, tokenizer_name: str) -> Tokenizer:
        # First try to find the tokenizer in the cache
        tokenizer: Optional[Tokenizer] = self.tokenizers.get(tokenizer_name)
        if tokenizer is not None:
            return tokenizer

        # Otherwise, create the tokenizer
        organization: str = tokenizer_name.split("/")[0]
        cache_config: CacheConfig = self.cache_backend_config.get_cache_config(organization)

        tokenizer_config = get_tokenizer_config(tokenizer_name)
        if tokenizer_config:
            tokenizer_spec = inject_object_spec_args(
                tokenizer_config.tokenizer_spec,
                constant_bindings={"cache_config": cache_config, "tokenizer_name": tokenizer_name},
                provider_bindings={
                    "api_key": lambda: provide_api_key(self.credentials, organization),
                    "project_id": lambda: self.credentials.get(organization + "ProjectId", None),  # VertexAI
                    "location": lambda: self.credentials.get(organization + "Location", None),  # VertexAI
                },
            )
            tokenizer = create_object(tokenizer_spec)
        else:
            hlog(f"No tokenizer config for {tokenizer_name}")

        # Cache the tokenizer
        assert isinstance(tokenizer, Tokenizer)  # To make mypy happy
        self.tokenizers[tokenizer_name] = tokenizer

        return tokenizer

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes based on the name of the tokenizer (e.g., huggingface/gpt2)."""

        @retry_tokenizer_request
        def tokenize_with_retry(tokenizer: Tokenizer, request: TokenizationRequest) -> TokenizationRequestResult:
            return tokenizer.tokenize(request)

        tokenizer: Tokenizer = self._get_tokenizer(request.tokenizer)

        try:
            return tokenize_with_retry(tokenizer=tokenizer, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to tokenize after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes based on the the name of the tokenizer (e.g., huggingface/gpt2)."""

        @retry_tokenizer_request
        def decode_with_retry(tokenizer: Tokenizer, request: DecodeRequest) -> DecodeRequestResult:
            return tokenizer.decode(request)

        tokenizer: Tokenizer = self._get_tokenizer(request.tokenizer)

        try:
            return decode_with_retry(tokenizer=tokenizer, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to decode after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")
