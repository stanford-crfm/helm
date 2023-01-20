import os
from dataclasses import replace
from typing import Dict, Optional

from retrying import RetryError, Attempt

from helm.common.cache import CacheConfig, MongoCacheConfig, SqliteCacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.proxy.retry import retry_request
from .client import Client
from .ai21_client import AI21Client
from .aleph_alpha_client import AlephAlphaClient
from .anthropic_client import AnthropicClient
from .chat_gpt_client import ChatGPTClient
from .cohere_client import CohereClient
from .together_client import TogetherClient
from .goose_ai_client import GooseAIClient
from .huggingface_client import HuggingFaceClient
from .ice_tokenizer_client import ICETokenizerClient
from .openai_client import OpenAIClient
from .microsoft_client import MicrosoftClient
from .perspective_api_client import PerspectiveAPIClient
from .yalm_tokenizer_client import YaLMTokenizerClient
from .simple_client import SimpleClient


class AutoClient(Client):
    """Automatically dispatch to the proper `Client` based on the organization."""

    def __init__(self, credentials: Dict[str, str], cache_path: str, mongo_uri: str = ""):
        self.credentials = credentials
        self.cache_path = cache_path
        self.mongo_uri = mongo_uri
        self.clients: Dict[str, Client] = {}
        self.tokenizer_clients: Dict[str, Client] = {}
        huggingface_cache_config = self._build_cache_config("huggingface")
        self.huggingface_client = HuggingFaceClient(huggingface_cache_config)
        hlog(f"AutoClient: cache_path = {cache_path}")
        hlog(f"AutoClient: mongo_uri = {mongo_uri}")

    def _build_cache_config(self, organization: str) -> CacheConfig:
        if self.mongo_uri:
            return MongoCacheConfig(self.mongo_uri, collection_name=organization)

        client_cache_path: str = os.path.join(self.cache_path, f"{organization}.sqlite")
        # TODO: Allow setting CacheConfig.follower_cache_path from a command line flag.
        return SqliteCacheConfig(client_cache_path)

    def get_client(self, request: Request) -> Client:
        """Return a client based on `organization`, creating it if necessary."""
        organization: str = request.model_organization
        client: Optional[Client] = self.clients.get(organization)

        if client is None:
            cache_config: CacheConfig = self._build_cache_config(organization)

            if organization == "openai":
                # TODO: add ChatGPT to the OpenAIClient when it's supported.
                #       We're using a separate client for now since we're using an unofficial Python library.
                # See https://github.com/acheong08/ChatGPT/wiki/Setup on how to get a valid session token.
                chat_gpt_client: ChatGPTClient = ChatGPTClient(
                    session_token=self.credentials.get("chatGPTSessionToken", ""),
                    lock_file_path=os.path.join(self.cache_path, "ChatGPT.lock"),
                    # TODO: use `cache_config` above. Since this feature is still experimental,
                    #       save queries and responses in a separate collection.
                    cache_config=self._build_cache_config("ChatGPT"),
                    tokenizer_client=self.get_tokenizer_client("huggingface"),
                )

                org_id = self.credentials.get("openaiOrgId", None)
                client = OpenAIClient(
                    api_key=self.credentials["openaiApiKey"],
                    cache_config=cache_config,
                    chat_gpt_client=chat_gpt_client,
                    org_id=org_id,
                )
            elif organization == "AlephAlpha":
                client = AlephAlphaClient(api_key=self.credentials["alephAlphaKey"], cache_config=cache_config)
            elif organization == "ai21":
                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_config=cache_config)
            elif organization == "cohere":
                client = CohereClient(api_key=self.credentials["cohereApiKey"], cache_config=cache_config)
            elif organization == "gooseai":
                org_id = self.credentials.get("gooseaiOrgId", None)
                client = GooseAIClient(
                    api_key=self.credentials["gooseaiApiKey"], cache_config=cache_config, org_id=org_id
                )
            elif organization == "huggingface":
                client = self.huggingface_client
            elif organization == "anthropic":
                client = AnthropicClient(api_key=self.credentials["anthropicApiKey"], cache_config=cache_config)
            elif organization == "microsoft":
                org_id = self.credentials.get("microsoftOrgId", None)
                lock_file_path: str = os.path.join(self.cache_path, f"{organization}.lock")
                client = MicrosoftClient(
                    api_key=self.credentials.get("microsoftApiKey", None),
                    lock_file_path=lock_file_path,
                    cache_config=cache_config,
                    org_id=org_id,
                )
            elif organization == "together":
                client = TogetherClient(api_key=self.credentials.get("togetherApiKey", None), cache_config=cache_config)
            elif organization == "simple":
                client = SimpleClient(cache_config=cache_config)
            else:
                raise ValueError(f"Unknown organization: {organization}")
            self.clients[organization] = client
        return client

    def make_request(self, request: Request) -> RequestResult:
        """
        Dispatch based on the organization in the name of the model (e.g., openai/davinci).
        Retries if request fails.
        """

        # TODO: need to revisit this because this swallows up any exceptions that are raised.
        @retry_request
        def make_request_with_retry(client: Client, request: Request) -> RequestResult:
            return client.make_request(request)

        organization: str = request.model_organization
        client: Client = self.get_client(request)

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
        client: Optional[Client] = self.tokenizer_clients.get(organization)

        if client is None:
            cache_config: CacheConfig = self._build_cache_config(organization)
            if organization in [
                "anthropic",
                "bigscience",
                "EleutherAI",
                "facebook",
                "google",
                "gooseai",
                "huggingface",
                "microsoft",
                "openai",
            ]:
                client = HuggingFaceClient(cache_config=cache_config)
            elif organization == "AlephAlpha":
                client = AlephAlphaClient(api_key=self.credentials["alephAlphaKey"], cache_config=cache_config)
            elif organization == "TsinghuaKEG":
                client = ICETokenizerClient(cache_config=cache_config)
            elif organization == "Yandex":
                client = YaLMTokenizerClient(cache_config=cache_config)
            elif organization == "ai21":
                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_config=cache_config)
            elif organization == "cohere":
                client = CohereClient(api_key=self.credentials["cohereApiKey"], cache_config=cache_config)
            elif organization == "simple":
                client = SimpleClient(cache_config=cache_config)
            else:
                raise ValueError(f"Unknown organization: {organization}")
            self.tokenizer_clients[organization] = client
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

    def get_toxicity_classifier_client(self) -> PerspectiveAPIClient:
        """Get the toxicity classifier client. We currently only support Perspective API."""
        cache_config: CacheConfig = self._build_cache_config("perspectiveapi")
        return PerspectiveAPIClient(self.credentials.get("perspectiveApiKey", ""), cache_config)
