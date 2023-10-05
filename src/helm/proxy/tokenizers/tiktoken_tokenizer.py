# mypy: check_untyped_defs = False
from typing import Any, Callable, Dict, List

from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from .cachable_tokenizer import CachableTokenizer

try:
    import tiktoken
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class TiktokenTokenizer(CachableTokenizer):
    def _tokenize_do_it(self, request: TokenizationRequest) -> Callable[[], Dict[str, Any]]:
        tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request.tokenizer))
        tokens = tokenizer.encode(request.text)
        if not request.encode:
            tokens = [tokenizer.decode([token]) for token in tokens]
            return {"token_strings": tokens}
        return {"token_ids": tokens}

    def _decode_do_it(self, request: DecodeRequest) -> Callable[[], Dict[str, Any]]:
        tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request.tokenizer))
        tokens = [token if isinstance(token, int) else tokenizer.encode(token)[0] for token in request.tokens]
        text = tokenizer.decode(tokens)
        return {"text": text}
