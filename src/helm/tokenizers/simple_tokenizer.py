from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
    TokenizationToken,
)
from helm.tokenizers.tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        if request.encode:
            return TokenizationRequestResult(
                success=True,
                cached=False,
                tokens=[TokenizationToken(ord(character)) for character in request.text],
                text=request.text,
            )
        else:
            return TokenizationRequestResult(
                success=True,
                cached=False,
                tokens=[TokenizationToken(character) for character in request.text],
                text=request.text,
            )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        return DecodeRequestResult(
            success=True, cached=False, text="".join([chr(code_point) for code_point in request.tokens])
        )
