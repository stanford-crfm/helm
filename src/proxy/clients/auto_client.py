import os
from dataclasses import replace
from typing import Dict, Optional

from retrying import RetryError, Attempt

from common.hierarchical_logger import hlog
from common.request import Request, RequestResult
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from proxy.retry import retry_request
from .client import Client
from .ai21_client import AI21Client
from .anthropic_client import AnthropicClient
from .together_client import TogetherClient
from .goose_ai_client import GooseAIClient
from .huggingface_client import HuggingFaceClient
from .openai_client import OpenAIClient
from .microsoft_client import MicrosoftClient
from .simple_client import SimpleClient


class AutoClient(Client):
    """Automatically dispatch to the proper `Client` based on the organization."""

    def __init__(self, credentials: Dict[str, str], cache_path: str):
        self.credentials = credentials
        self.cache_path = cache_path
        self.clients: Dict[str, Client] = {}
        self.huggingface_client = HuggingFaceClient(cache_path=os.path.join(self.cache_path, "huggingface.sqlite"))

    def get_client(self, organization: str) -> Client:
        """Return a client based on `organization`, creating it if necessary."""
        client: Optional[Client] = self.clients.get(organization)

        if client is None:
            org_id: Optional[str]
            client_cache_path: str = os.path.join(self.cache_path, f"{organization}.sqlite")

            if organization == "openai":
                org_id = self.credentials.get("openaiOrgId", None)
                client = OpenAIClient(
                    api_key=self.credentials["openaiApiKey"], cache_path=client_cache_path, org_id=org_id
                )
            elif organization == "ai21":
                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_path=client_cache_path)
            elif organization == "gooseai":
                org_id = self.credentials.get("gooseaiOrgId", None)
                client = GooseAIClient(
                    api_key=self.credentials["gooseaiApiKey"], cache_path=client_cache_path, org_id=org_id
                )
            elif organization == "huggingface":
                client = self.huggingface_client
            elif organization == "anthropic":
                client = AnthropicClient(api_key=self.credentials["anthropicApiKey"], cache_path=client_cache_path)
            elif organization == "microsoft":
                org_id = self.credentials.get("microsoftOrgId", None)
                client = MicrosoftClient(
                    api_key=self.credentials["microsoftApiKey"], cache_path=client_cache_path, org_id=org_id
                )
            elif organization == "together":
                client = TogetherClient(cache_path=client_cache_path)
            elif organization == "simple":
                client = SimpleClient(cache_path=client_cache_path)
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

    def get_tokenizer_client(self, organization: str) -> Client:
        """Return a client based on `organization`, creating it if necessary."""
        client: Optional[Client] = self.clients.get(organization)

        if client is None:
            client_cache_path: str = os.path.join(self.cache_path, f"{organization}.sqlite")
            if organization in [
                "anthropic",
                "bigscience",
                "EleutherAI",
                "google",
                "gooseai",
                "huggingface",
                "microsoft",
                "openai",
            ]:
                client = HuggingFaceClient(cache_path=client_cache_path)
            elif organization == "ai21":
                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_path=client_cache_path)
            elif organization == "simple":
                client = SimpleClient(cache_path=client_cache_path)
            else:
                raise ValueError(f"Unknown organization: {organization}")
            self.clients[organization] = client
        return client

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes based on the organization in the name of the tokenizer (e.g., huggingface/gpt2)."""

        @retry_request
        def tokenize_with_retry(client: Client, request: TokenizationRequest) -> TokenizationRequestResult:
            return client.tokenize(request)

        organization: str = request.tokenizer_organization
        client: Client = self.get_tokenizer_client(organization)

        try:
            return tokenize_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to tokenize after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes based on the organization in the name of the tokenizer (e.g., huggingface/gpt2)."""

        @retry_request
        def decode_with_retry(client: Client, request: DecodeRequest) -> DecodeRequestResult:
            return client.decode(request)

        organization: str = request.tokenizer_organization
        client: Client = self.get_tokenizer_client(organization)

        try:
            return decode_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to decode after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")
