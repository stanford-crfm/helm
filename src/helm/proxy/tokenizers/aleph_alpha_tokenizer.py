# mypy: check_untyped_defs = False
from typing import Any, Dict, List

from tokenizers import Tokenizer as InternalTokenizer, Encoding

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from .caching_tokenizer import CachingTokenizer

try:
    from aleph_alpha_client import Client as AlephAlphaPythonClient
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


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
        self._aleph_alpha_client = AlephAlphaPythonClient(token=api_key)
        self._tokenizer_name_to_tokenizer: Dict[str, InternalTokenizer] = {}

    def _get_tokenizer(self, tokenizer_name: str) -> InternalTokenizer:
        if tokenizer_name not in self.VALID_TOKENIZERS:
            raise ValueError(f"Invalid tokenizer: {tokenizer_name}")

        # Check if the tokenizer is cached
        if tokenizer_name not in self._tokenizer_name_to_tokenizer:
            self._tokenizer_name_to_tokenizer[tokenizer_name] = self._aleph_alpha_client.tokenizer(tokenizer_name)
            hlog(f"Initialized tokenizer: {tokenizer_name}")
        return self._tokenizer_name_to_tokenizer[tokenizer_name]

    @property
    def use_encode_in_cache_key(self) -> bool:
        """Since the tokenization endpoint returns both the token IDs and the token strings, we don't need to
        use the `encode` parameter in the cache key.
        """
        return False

    def _post_process_decoding(self, text: str, request: DecodeRequest) -> str:
        # The text always seems to start with a single whitespace when encoding/decoding.
        return text.replace(" ", "", 1)

    def _tokenize_do_it(self, request: TokenizationRequest) -> Dict[str, Any]:
        tokenizer: InternalTokenizer = self._get_tokenizer(request.tokenizer_name)
        result: Encoding = tokenizer.encode(request.text, add_special_tokens=False)
        return {"token_ids": result.ids, "token_strings": result.tokens}

    def _decode_do_it(self, request: DecodeRequest) -> Dict[str, Any]:
        tokenizer: InternalTokenizer = self._get_tokenizer(request.tokenizer_name)
        text = tokenizer.decode(request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces)
        return {"text": text}
