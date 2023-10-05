# mypy: check_untyped_defs = False
from typing import Any, Callable, Dict

from icetk import icetk as icetk_tokenizer

from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from .cachable_tokenizer import CachableTokenizer
from .tokenizer import cleanup_tokens


class ICETokenizer(CachableTokenizer):
    def _tokenize_do_it(self, request: TokenizationRequest) -> Callable[[], Dict[str, Any]]:
        tokens = icetk_tokenizer.encode(request.text) if request.encode else icetk_tokenizer.tokenize(request.text)
        if not request.encode:
            tokens = cleanup_tokens(tokens, request.tokenizer)
            return {"token_strings": tokens}
        return {"token_ids": tokens}

    def _decode_do_it(self, request: DecodeRequest) -> Callable[[], Dict[str, Any]]:
        text = icetk_tokenizer.decode(request.tokens)
        return {"text": text}
