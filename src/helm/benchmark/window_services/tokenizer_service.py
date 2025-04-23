from helm.common.context import Context
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)


# TODO: Rename this to TokenizerContext
class TokenizerService:
    """
    A wrapper around `Context` that makes only necessary server requests to tokenize.
    """

    def __init__(self, context: Context):
        self._context: Context = context

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenize via an API."""
        return self._context.tokenize(request)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decode via an API."""
        return self._context.decode(request)
