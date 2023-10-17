# mypy: check_untyped_defs = False
from typing import Any, Dict, List

from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from .caching_tokenizer import CachingTokenizer

try:
    import tiktoken
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class TiktokenTokenizer(CachingTokenizer):
    def _tokenize_do_it(self, request: TokenizationRequest) -> Dict[str, Any]:
        tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request.tokenizer))
        token_ids: List[int] = tokenizer.encode(request.text)
        if not request.encode:
            token_strings = [tokenizer.decode([token]) for token in token_ids]
            return {"token_strings": token_strings}
        return {"token_ids": token_ids}

    def _decode_do_it(self, request: DecodeRequest) -> Dict[str, Any]:
        tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request.tokenizer))
        # TODO: This is done to support deconding of token strings, but it should not
        # be needed as a decode request should only contain token ids.
        tokens = [token if isinstance(token, int) else tokenizer.encode(str(token))[0] for token in request.tokens]
        text = tokenizer.decode(tokens)
        return {"text": text}
