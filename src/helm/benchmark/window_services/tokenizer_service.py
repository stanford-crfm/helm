from helm.common.authentication import Authentication
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.proxy.services.service import Service


class TokenizerService:
    """
    A wrapper around `Service` that makes only necessary server requests to tokenize.
    """

    def __init__(self, service: Service, auth: Authentication):
        self._service: Service = service
        self._auth: Authentication = auth

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenize via an API."""
        return self._service.tokenize(self._auth, request)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decode via an API."""
        return self._service.decode(self._auth, request)
