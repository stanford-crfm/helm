# mypy: check_untyped_defs = False
from typing import Any, Callable, Dict, List

from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from .tokenizer import Tokenizer

try:
    import tiktoken
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class TiktokenTokenizer(Tokenizer):
    @property
    def supported_tokenizers(self) -> List[str]:
        return ["openai/cl100k_base"]

    def _get_tokenize_do_it(self, request: TokenizationRequest) -> Callable[[], Dict[str, Any]]:
        def do_it():
            tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request.tokenizer))
            tokens = tokenizer.encode(request.text)
            if not request.encode:
                tokens = [tokenizer.decode([token]) for token in tokens]
            if request.truncation:
                tokens = tokens[: request.max_length]
            return {"tokens": tokens}

        return do_it

    def _get_decode_do_it(self, request: DecodeRequest) -> Callable[[], Dict[str, Any]]:
        def do_it():
            tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request.tokenizer))
            tokens = [token if isinstance(token, int) else tokenizer.encode(token)[0] for token in request.tokens]
            text = tokenizer.decode(tokens)
            return {"text": text}

        return do_it
