import dataclasses
import os
from typing import Any, Dict, List, Optional

import requests

from helm.common.cache import CacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationToken,
)
from helm.tokenizers.caching_tokenizer import CachingTokenizer


class GrokAPITokenizer(CachingTokenizer):
    """Tokenizer that uses the xAI Grok Tokenize Text API

    Doc: https://docs.x.ai/docs/api-reference#tokenize-text"""

    def __init__(self, cache_config: CacheConfig, api_key: Optional[str] = None) -> None:
        super().__init__(cache_config)
        self.api_key = api_key or os.environ.get("XAI_API_KEY")

    def _tokenization_request_to_cache_key(self, request: TokenizationRequest) -> Dict[str, Any]:
        cache_key = dataclasses.asdict(request)
        # Delete encode because the Grok API simulateously gives string and integer tokens.
        del cache_key["encode"]
        return cache_key

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise Exception(
                "No Grok API key found. "
                "Set grokApiKey in credentials.conf or set the GROK_API_KEY environment variable"
            )
        text = request["text"]
        model = request["tokenizer"].split("/")[-1]
        response = requests.post(
            url="https://api.x.ai/v1/tokenize-text",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"text": text, "model": model},
        )
        response.raise_for_status()
        return response.json()

    def _tokenization_raw_response_to_tokens(
        self, response: Dict[str, Any], request: TokenizationRequest
    ) -> List[TokenizationToken]:
        raw_token_field_name = "token_id" if request.encode else "string_token"
        return [TokenizationToken(raw_token[raw_token_field_name]) for raw_token in response["token_ids"]]

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("The xAI API does not support decoding.")
