from typing import List, Dict

from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from .tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    @staticmethod
    def tokenize_by_space(text: str) -> List[str]:
        """Simply tokenizes by a single white space."""
        return text.split(" ")

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        if request.tokenizer == "simple/model1":
            raw_tokens: List[str] = SimpleTokenizer.tokenize_by_space(request.text)
            return TokenizationRequestResult(
                success=True, cached=False, tokens=[TokenizationToken(text) for text in raw_tokens], text=request.text
            )
        else:
            raise ValueError("Unknown model")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        text = " ".join([token.text for token in request.tokens])
        return DecodeRequestResult(success=True, cached=False, text=text)
