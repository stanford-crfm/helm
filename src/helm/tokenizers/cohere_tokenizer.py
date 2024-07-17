from typing import Any, Dict, List, Optional

import cohere
from cohere.manually_maintained.tokenizers import get_hf_tokenizer

from helm.common.cache import CacheConfig
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationToken,
)
from helm.tokenizers.caching_tokenizer import CachingTokenizer


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
