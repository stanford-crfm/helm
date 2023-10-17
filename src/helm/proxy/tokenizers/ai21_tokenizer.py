from typing import Dict, List
import requests

from dacite import from_dict

from helm.common.cache import Cache, CacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
    TextRange,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.proxy.clients.ai21_utils import AI21RequestError, handle_failed_request
from .tokenizer import Tokenizer


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

        def do_it():
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
