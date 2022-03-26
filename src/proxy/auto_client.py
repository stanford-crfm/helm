import os
from typing import Dict

from common.request import Request, RequestResult
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from .client import Client
from .ai21_client import AI21Client
from .huggingface_client import HuggingFaceClient
from .openai_client import OpenAIClient
from .simple_client import SimpleClient


class AutoClient(Client):
    """Automatically dispatch to the proper `Client` based on the organization."""

    def __init__(self, credentials: Dict[str, str], cache_path: str):
        self.credentials = credentials
        self.cache_path = cache_path
        self.clients: Dict[str, Client] = {}

    def get_client(self, organization: str) -> Client:
        """Return a client based on `organization`, creating it if necessary."""
        client = self.clients.get(organization)
        if client is None:
            client_cache_path = os.path.join(self.cache_path, f"{organization}.sqlite")
            if organization == "openai":
                client = OpenAIClient(api_key=self.credentials["openaiApiKey"], cache_path=client_cache_path)
            elif organization == "ai21":
                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_path=client_cache_path)
            elif organization == "huggingface":
                client = HuggingFaceClient(cache_path=client_cache_path)
            elif organization == "simple":
                client = SimpleClient()
            else:
                raise Exception(f"Unknown organization: {organization}")
            self.clients[organization] = client
        return client

    def make_request(self, request: Request) -> RequestResult:
        """Dispatch based on the organization in the name of the model (e.g., openai/davinci)."""
        organization: str = request.model_organization
        client: Client = self.get_client(organization)
        return client.make_request(request)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes based on the organization in the name of the model (e.g., ai21/j1-jumbo)."""
        organization: str = request.model_organization
        client: Client = self.get_client(organization)
        return client.tokenize(request)
