from typing import List, Optional

from helm.benchmark.window_services.window_service import EncodeResult
from helm.benchmark.window_services.default_window_service import DefaultWindowService
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)


class NoDecodingWindowService(DefaultWindowService):
    """A window service for tokenizers that have a unimplemented decode() method.

    This assumes that concatenating the tokens from the tokenize endpoint will result in the original string,
    which is not always true for all tokenizers.

    TODO(#2141): Come up with a more correct way of doing this."""

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        response: TokenizationRequestResult = self.service.tokenize(
            TokenizationRequest(text, tokenizer=self.tokenizer_name, encode=False, truncation=truncation)
        )
        return EncodeResult(text=text, tokens=response.tokens[:max_length])

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        del normalized_text
        token_strings = []
        for token in tokens:
            assert isinstance(token.value, str)
            token_strings.append(token.value)
        return "".join(token_strings)
