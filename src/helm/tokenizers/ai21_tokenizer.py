import threading
from typing import Any, Dict, List
import requests

from dacite import from_dict

from helm.common.cache import Cache, CacheConfig
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
    TextRange,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.clients.ai21_utils import AI21RequestError, handle_failed_request
from helm.tokenizers.caching_tokenizer import CachingTokenizer
from helm.tokenizers.tokenizer import Tokenizer

try:
    from ai21_tokenizer import Tokenizer as SDKTokenizer, PreTrainedTokenizers
    from ai21_tokenizer.base_tokenizer import BaseTokenizer
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["ai21"])


class AI21Tokenizer(Tokenizer):
    def __init__(self, api_key: str, cache_config: CacheConfig) -> None:
        self.cache = Cache(cache_config)
        self.api_key: str = api_key

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """
        Tokenizes the text by using the AI21 endpoint: https://api.ai21.com/studio/v1/tokenize.
        """
        # TODO: Does not support encoding
        raw_request: Dict[str, str] = {"text": request.text}

        def do_it() -> Dict[str, Any]:
            response = requests.post(
                "https://api.ai21.com/studio/v1/tokenize",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=raw_request,
            ).json()

            # If 'tokens' is not present in the response, assume request failed.
            if "tokens" not in response:
                handle_failed_request(api_type="tokenizer", response=response)

            return response

        try:
            response, cached = self.cache.get(raw_request, do_it)
        except AI21RequestError:
            return TokenizationRequestResult(success=False, cached=False, text="", tokens=[])

        # Each token is represented like this in the response:
        # {'token': '▁Hello', 'textRange': {'start': 0, 'end': 5}}
        tokens: List[TokenizationToken] = []
        for token_dict in response["tokens"]:
            tokens.append(
                TokenizationToken(value=token_dict["token"], text_range=from_dict(TextRange, token_dict["textRange"]))
            )
        text: str = response["text"]
        return TokenizationRequestResult(success=True, cached=cached, tokens=tokens, text=text)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Not supported")


class AI21LocalTokenizer(CachingTokenizer):
    """AI21 tokenizer using the AI21 Python library."""

    _KNOWN_TOKENIZERS = [
        PreTrainedTokenizers.J2_TOKENIZER,
        PreTrainedTokenizers.JAMBA_TOKENIZER,
        PreTrainedTokenizers.JAMBA_INSTRUCT_TOKENIZER,
    ]

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        self._tokenizers_lock = threading.Lock()
        self.tokenizers: Dict[str, BaseTokenizer] = {}

    def _get_tokenizer(self, tokenizer_name: str) -> BaseTokenizer:
        if tokenizer_name not in AI21LocalTokenizer._KNOWN_TOKENIZERS:
            raise ValueError(
                f"Unknown tokenizer: {tokenizer_name} - " f"valid values are {AI21LocalTokenizer._KNOWN_TOKENIZERS}"
            )
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
