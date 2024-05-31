import json
import requests
from typing import Any, Dict, List, Optional

import cohere
from cohere.manually_maintained.tokenizers import get_hf_tokenizer

from helm.common.cache import CacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from helm.clients.cohere_utils import get_cohere_url, DEFAULT_COHERE_API_VERSION
from helm.tokenizers.caching_tokenizer import CachingTokenizer


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

    def _tokenization_request_to_cache_key(self, request: TokenizationRequest) -> Dict[str, Any]:
        # This cache key is used to preserve our existing Cache (10/17/2023)
        return {"text": request.text}

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the request to the Cohere Tokenize API.

        From https://docs.cohere.ai/tokenize-reference, for text "tokenize me! :D", the response will be:

        {
            "tokens": [34160, 974, 514, 34, 1420, 69]
            "token_strings": ["token", "ize", " me", "!", " :", "D"]
        }
        """
        text: str = request["text"]
        assert (
            1 <= len(text) <= CohereTokenizer.TOKENIZE_API_MAX_TEXT_LENGTH
        ), f"Invalid text length: {len(text)}. Valid length: [1..{CohereTokenizer.TOKENIZE_API_MAX_TEXT_LENGTH:,d}]"

        response = requests.request(
            method="POST",
            url=get_cohere_url(CohereTokenizer.TOKENIZE_ENDPOINT),
            headers={
                "Authorization": f"BEARER {self.api_key}",
                "Content-Type": "application/json",
                "Cohere-Version": DEFAULT_COHERE_API_VERSION,
            },
            data=json.dumps(request),
        )
        result = json.loads(response.text)
        assert "message" not in result.keys(), f"Request failed with error {result['message']}"
        assert "tokens" in result and "token_strings" in result, f"Invalid response: {result}"
        # This output format is used to preserve our existing Cache (10/17/2023)
        return result

    def _tokenization_raw_response_to_tokens(
        self, response: Dict[str, Any], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        tokens = response["tokens" if request.encode else "token_strings"]
        return [TokenizationToken(token) for token in tokens]

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Defined for mypy but decode() already raises NotImplementedError
        raise NotImplementedError("The Cohere API does not support decoding.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("The Cohere API does not support decoding.")


class CohereLocalTokenizer(CachingTokenizer):
    """Cohere tokenizer using the Cohere Python library."""

    def __init__(self, api_key: Optional[str], cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        self.client = cohere.Client(api_key)

    def _tokenization_request_to_cache_key(self, request: TokenizationRequest) -> Dict[str, Any]:
        return {"text": request.text, "tokenizer": request.tokenizer}

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        model: str = request["tokenizer"].split("/")[1]
        # Workaround for https://github.com/cohere-ai/cohere-python/issues/493
        # `token_strings` are always set to `[]`, so we have to populate it ourselves.
        response = self.client.tokenize(text=request["text"], model=model)
        response_dict = response.dict()
        response_dict["token_strings"] = get_hf_tokenizer(self.client, model).decode_batch(
            [[token] for token in response.tokens]
        )
        return response_dict

    def _tokenization_raw_response_to_tokens(
        self, response: Dict[str, Any], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        tokens: List[TokenizationToken] = []
        if request.encode:
            tokens = [TokenizationToken(token) for token in response["tokens"]]
        else:
            tokens = [TokenizationToken(token) for token in response["token_strings"]]
        if request.truncation:
            tokens = tokens[: request.max_length]
        return tokens

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        model: str = request["tokenizer"].split("/")[1]
        response = self.client.detokenize(tokens=request["tokens"], model=model)
        return response.dict()
