import os
from dataclasses import replace
from typing import Dict

from retrying import RetryError, Attempt

from common.hierarchical_logger import hlog
from common.request import Request, RequestResult
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from .client import Client
from .ai21_client import AI21Client
from .anthropic_client import AnthropicClient
from .huggingface_client import HuggingFaceClient
from .openai_client import OpenAIClient
from .microsoft_client import MicrosoftClient
from .simple_client import SimpleClient
from .retry import retry_request


class AutoClient(Client):
    """Automatically dispatch to the proper `Client` based on the organization."""

    def __init__(self, credentials: Dict[str, str], cache_path: str):
        self.credentials = credentials
        self.cache_path = cache_path
        self.clients: Dict[str, Client] = {}

    def get_client(self, organization: str) -> Client:
        """Return a client based on `organization`, creating it if necessary."""
        client: Client = self.clients.get(organization)

        if client is None:
            client_cache_path: str = os.path.join(self.cache_path, f"{organization}.sqlite")
            if organization == "openai":
                client = OpenAIClient(api_key=self.credentials["openaiApiKey"], cache_path=client_cache_path)
            elif organization == "ai21":
                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_path=client_cache_path)
            elif organization == "huggingface":
                client = HuggingFaceClient(cache_path=client_cache_path)
            elif organization == "anthropic":
                client = AnthropicClient(api_key=self.credentials["anthropicApiKey"], cache_path=client_cache_path)
            elif organization == "microsoft":
                client = MicrosoftClient(api_key=self.credentials["microsoftApiKey"], cache_path=client_cache_path)
            elif organization == "simple":
                client = SimpleClient()
            else:
                raise ValueError(f"Unknown organization: {organization}")
            self.clients[organization] = client
        return client

    def make_request(self, request: Request) -> RequestResult:
        """
        Dispatch based on the organization in the name of the model (e.g., openai/davinci).
        Retries if request fails.
        """

        @retry_request
        def make_request_with_retry(client: Client, request: Request) -> RequestResult:
            return client.make_request(request)

        organization: str = request.model_organization
        client: Client = self.get_client(organization)

        try:
            return make_request_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = (
                f"Failed to make request to {organization} after retrying {last_attempt.attempt_number} times"
            )
            hlog(retry_error)

            # Notify our user that we failed to make the request even after retrying.
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes based on the organization in the name of the model (e.g., ai21/j1-jumbo)."""
        organization: str = request.model_organization
        client: Client = self.get_client(organization)
        return client.tokenize(request)
