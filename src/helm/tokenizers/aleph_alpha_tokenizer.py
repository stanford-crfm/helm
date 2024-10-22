from typing import Any, Dict, List

from tokenizers import Tokenizer as InternalTokenizer, Encoding

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
    TokenizationToken,
)
from helm.tokenizers.caching_tokenizer import CachingTokenizer

try:
    from aleph_alpha_client import Client as AlephAlphaPythonClient
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["aleph-alpha"])


class AlephAlphaTokenizer(CachingTokenizer):
    TOKENIZE_ENDPOINT: str = "tokenize"
    DETOKENIZE_ENDPOINT: str = "detokenize"

    VALID_TOKENIZERS: List[str] = [
        "luminous-base",
        "luminous-extended",
        "luminous-supreme",
    ]

    def __init__(self, api_key: str, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        self.api_key: str = api_key
        self._aleph_alpha_client = AlephAlphaPythonClient(token=api_key) if api_key else None
        self._tokenizer_name_to_tokenizer: Dict[str, InternalTokenizer] = {}

    def _get_tokenizer(self, tokenizer_name: str) -> InternalTokenizer:
        if tokenizer_name not in self.VALID_TOKENIZERS:
            raise ValueError(f"Invalid tokenizer: {tokenizer_name}")

        # Check if the tokenizer is cached
        if tokenizer_name not in self._tokenizer_name_to_tokenizer:
            if self._aleph_alpha_client is None:
                raise ValueError("Aleph Alpha API key not set.")
            self._tokenizer_name_to_tokenizer[tokenizer_name] = self._aleph_alpha_client.tokenizer(tokenizer_name)
            hlog(f"Initialized tokenizer: {tokenizer_name}")
        return self._tokenizer_name_to_tokenizer[tokenizer_name]

    def _tokenization_request_to_cache_key(self, request: TokenizationRequest) -> Dict[str, Any]:
        # This cache key is used to preserve our existing Cache (10/17/2023)
        cache_key: Dict[str, Any] = {
            "model": request.tokenizer_name,
            "prompt": request.text,
            "tokens": True,
            "token_ids": True,
        }
        return cache_key

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer: InternalTokenizer = self._get_tokenizer(request["model"])
        result: Encoding = tokenizer.encode(request["prompt"], add_special_tokens=False)
        # This output using "token_ids" and "tokens" is used to preserve our existing Cache (10/17/2023)
        return {"token_ids": result.ids, "tokens": result.tokens}

    def _tokenization_raw_response_to_tokens(
        self, response: Dict[str, Any], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        tokens: list = response["token_ids" if request.encode else "tokens"]
        return [TokenizationToken(token) for token in tokens]

    def _decode_request_to_cache_key(self, request: DecodeRequest) -> Dict[str, Any]:
        # This cache key is used to preserve our existing Cache (10/17/2023)
        cache_key: Dict[str, Any] = {
            "model": request.tokenizer_name,
            "token_ids": request.tokens,
        }
        return cache_key

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer: InternalTokenizer = self._get_tokenizer(request["model"])
        text = tokenizer.decode(request["token_ids"])
        # This output using "result" is used to preserve our existing Cache (10/17/2023)
        return {"result": text}

    def _decode_raw_response_to_text(self, response: Dict[str, Any], request: DecodeRequest) -> str:
        # The text always seems to start with a single whitespace when encoding/decoding.
        return response["result"].replace(" ", "", 1)
