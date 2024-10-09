import threading
from typing import Any, Dict

from helm.common.cache import CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.tokenizers.caching_tokenizer import CachingTokenizer

try:
    from ai21_tokenizer import Tokenizer as SDKTokenizer
    from ai21_tokenizer.base_tokenizer import BaseTokenizer
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["ai21"])


class AI21LocalTokenizer(CachingTokenizer):
    """AI21 tokenizer using the AI21 Python library."""

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        self._tokenizers_lock = threading.Lock()
        self.tokenizers: Dict[str, BaseTokenizer] = {}

    def _get_tokenizer(self, tokenizer_name: str) -> BaseTokenizer:
        with self._tokenizers_lock:
            if tokenizer_name not in self.tokenizers:
                self.tokenizers[tokenizer_name] = SDKTokenizer.get_tokenizer(tokenizer_name)
            return self.tokenizers[tokenizer_name]

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer_name = request["tokenizer"].split("/")[1]
        tokenizer = self._get_tokenizer(tokenizer_name)
        if request["truncation"]:
            token_ids = tokenizer.encode(
                text=request["text"],
                truncation=request["truncation"],
                max_length=request["max_length"],
                add_special_tokens=False,
            )
        else:
            token_ids = tokenizer.encode(
                text=request["text"],
                add_special_tokens=False,
            )
        if request["encode"]:
            return {"tokens": token_ids}
        else:
            return {"tokens": tokenizer.convert_ids_to_tokens(token_ids)}

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer_name = request["tokenizer"].split("/")[1]
        tokenizer = self._get_tokenizer(tokenizer_name)
        return {"text": tokenizer.decode(request["tokens"])}
