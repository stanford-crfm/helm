import json
import requests
from typing import Any, Dict

from helm.common.cache import CacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.proxy.clients.cohere_utils import get_cohere_url, DEFAULT_COHERE_API_VERSION
from .caching_tokenizer import CachingTokenizer


class CohereTokenizer(CachingTokenizer):
    # From "https://docs.cohere.ai/versioning-reference",
    # "this version [2021-11-08] introduces multiple generations, meaning that the generations endpoint will
    # now accept a num_generations argument in the JSON and will always return an array of generations"
    # Note that the API version is decoupled from the model version.
    DEFAULT_API_VERSION: str = "2021-11-08"

    TOKENIZE_ENDPOINT: str = "tokenize"

    # According to https://docs.cohere.ai/tokenize-reference#request, for tokenize, text: "the string to
    # be tokenized, the minimum text length is 1 character, and the maximum text length is 65536 characters."
    # However, even sending a request with 60,000 characters sometimes fails, so we set the
    # maximum length to 50,000, which is about 8,333 tokens.
    # TODO: followed up with Cohere support with an example of a failure case
    TOKENIZE_API_MAX_TEXT_LENGTH: int = 50_000

    def __init__(self, api_key: str, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        self.api_key: str = api_key

    @property
    def use_encode_in_cache_key(self) -> bool:
        """Since the tokenization endpoint returns both the token IDs and the token strings, we don't need to
        use the `encode` parameter in the cache key.
        """
        return False

    def _tokenize_do_it(self, request: TokenizationRequest) -> Dict[str, Any]:
        """
        Send the request to the Cohere Tokenize API.

        From https://docs.cohere.ai/tokenize-reference, for text "tokenize me! :D", the response will be:

        {
            "tokens": [34160, 974, 514, 34, 1420, 69]
            "token_strings": ["token", "ize", " me", "!", " :", "D"]
        }
        """
        text: str = request.text
        assert (
            1 <= len(text) <= CohereTokenizer.TOKENIZE_API_MAX_TEXT_LENGTH
        ), f"Invalid text length: {len(text)}. Valid length: [1..{CohereTokenizer.TOKENIZE_API_MAX_TEXT_LENGTH:,d}]"
        raw_request: Dict[str, str] = {"text": text}

        response = requests.request(
            method="POST",
            url=get_cohere_url(CohereTokenizer.TOKENIZE_ENDPOINT),
            headers={
                "Authorization": f"BEARER {self.api_key}",
                "Content-Type": "application/json",
                "Cohere-Version": DEFAULT_COHERE_API_VERSION,
            },
            data=json.dumps(raw_request),
        )
        result = json.loads(response.text)
        assert "message" not in result.keys(), f"Request failed with error {result['message']}"
        assert "tokens" in result and "token_strings" in result, f"Invalid response: {result}"
        return {"token_ids": result["tokens"], "token_strings": result["token_strings"]}

    def _decode_do_it(self, request: DecodeRequest) -> Dict[str, Any]:
        pass

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("The Cohere API does not support decoding.")
