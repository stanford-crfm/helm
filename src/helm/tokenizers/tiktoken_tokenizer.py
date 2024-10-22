from typing import Any, Dict

from helm.common.optional_dependencies import handle_module_not_found_error
from helm.tokenizers.caching_tokenizer import CachingTokenizer

try:
    import tiktoken
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class TiktokenTokenizer(CachingTokenizer):
    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request["tokenizer"]))
        tokens = tokenizer.encode(request["text"])
        if not request["encode"]:
            tokens = [tokenizer.decode([token]) for token in tokens]  # type: ignore
        if request["truncation"]:
            tokens = tokens[: request["max_length"]]
        return {"tokens": tokens}

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer = tiktoken.get_encoding(self._get_tokenizer_name(request["tokenizer"]))
        # TODO: This is done to support decoding of token strings, but it should not
        # be needed as a decode request should only contain token ids.
        tokens = [token if isinstance(token, int) else tokenizer.encode(token)[0] for token in request["tokens"]]
        text = tokenizer.decode(tokens)
        return {"text": text}
