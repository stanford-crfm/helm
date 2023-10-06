# mypy: check_untyped_defs = False
from typing import Any, Dict

from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from helm.common.optional_dependencies import handle_module_not_found_error
from .cachable_tokenizer import CachableTokenizer
from .tokenizer import cleanup_tokens

try:
    from icetk import icetk as icetk_tokenizer
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class ICETokenizer(CachableTokenizer):
    def _tokenize_do_it(self, request: TokenizationRequest) -> Dict[str, Any]:
        tokens = icetk_tokenizer.encode(request.text) if request.encode else icetk_tokenizer.tokenize(request.text)
        if not request.encode:
            tokens = cleanup_tokens(tokens, request.tokenizer)
            return {"token_strings": tokens}
        return {"token_ids": tokens}

    def _decode_do_it(self, request: DecodeRequest) -> Dict[str, Any]:
        text = icetk_tokenizer.decode(request.tokens)
        return {"text": text}
