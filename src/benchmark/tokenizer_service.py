from abc import ABC, abstractmethod

from common.authentication import Authentication
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from proxy.service import Service


class TokenizerService(ABC):
    """
    A wrapper around `Service` that makes only necessary server requests to tokenize.
    """

    @abstractmethod
    def __init__(self, service: Service, auth: Authentication):
        self._service: Service = service
        self._auth: Authentication = auth

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenize via an API."""
        return self._service.tokenize(self._auth, request)
