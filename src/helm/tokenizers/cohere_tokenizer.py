import json
import requests
from typing import Any, Dict, List

from helm.common.cache import CacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from helm.clients.cohere_utils import get_cohere_url
from .caching_tokenizer import CachingTokenizer


class CohereTokenizer(CachingTokenizer):
    TOKENIZE_ENDPOINT: str = "tokenize"

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
        assert (1 <= len(text)), f"Invalid empty text with length: {len(text)}. Valid length: >= 1"
        request["model"] = "command-r" ### To do

        response = requests.request(
            method="POST",
            url=get_cohere_url(CohereTokenizer.TOKENIZE_ENDPOINT),
            headers={
                "Authorization": f"BEARER {self.api_key}",
                "Content-Type": "application/json",
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
