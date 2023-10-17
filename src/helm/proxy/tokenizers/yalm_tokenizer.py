# mypy: check_untyped_defs = False
from typing import Any, Dict

from helm.common.cache import CacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from .caching_tokenizer import CachingTokenizer
from .tokenizer import cleanup_tokens
from .yalm_tokenizer_data.yalm_tokenizer import YaLMTokenizer as YaLMTokenizerInternal


class YaLMTokenizer(CachingTokenizer):
    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        self._tokenizer = YaLMTokenizerInternal()

    def _tokenize_do_it(self, request: TokenizationRequest) -> Dict[str, Any]:
        token_ids = self._tokenizer.tokenize(request.text)
        if request.truncation:
            token_ids = token_ids[: request.max_length]
        # We do not use:
        # return {"tokens": token_ids if request.encode else self._tokenizer.convert_ids_to_string(token_ids)}
        # as this replace "▁" with an empty string, which is not what we want.
        # This is a problem because then tokenize(" Hello", encode=False) == tokenize("Hello", encode=False)
        # That is why we manually replace "▁" with a space.
        if request.encode:
            return {"token_ids": token_ids}

        # No encoding, just return the token strings
        tokens = cleanup_tokens(self._tokenizer.convert_ids_to_tokens(token_ids), request.tokenizer)
        return {"token_strings": tokens}

    def _decode_do_it(self, request: DecodeRequest) -> Dict[str, Any]:
        text = self._tokenizer.decode(request.tokens)
        return {"text": text}
