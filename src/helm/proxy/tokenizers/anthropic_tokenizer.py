# mypy: check_untyped_defs = False
from typing import Any, Callable, Dict, List

import threading

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
)
from .cachable_tokenizer import CachableTokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

try:
    import anthropic
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class AnthropicTokenizer(CachableTokenizer):
    LOCK: threading.Lock = threading.Lock()

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        with AnthropicTokenizer.LOCK:
            self._tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(
                tokenizer_object=anthropic.get_tokenizer()
            )

    def _tokenize_do_it(self, request: TokenizationRequest) -> Callable[[], Dict[str, Any]]:
        if request.encode:
            if request.truncation:
                tokens = self._tokenizer.encode(
                    request.text,
                    truncation=request.truncation,
                    max_length=request.max_length,
                    add_special_tokens=False,
                )
            else:
                tokens = self._tokenizer.encode(request.text, add_special_tokens=False)
            return {"token_ids": tokens}

        # No encoding, just return the token strings
        tokens = [self._tokenizer.convert_tokens_to_string([i]) for i in self.tokenizer.tokenize(request.text)]
        return {"token_strings": tokens}

    def _decode_do_it(self, request: DecodeRequest) -> Callable[[], Dict[str, Any]]:
        text = self._tokenizer.decode(request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces)
        return {"text": text}
