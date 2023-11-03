import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from retrying import Attempt, RetryError

from helm.benchmark.model_deployment_registry import get_model_deployment
from helm.benchmark.tokenizer_config_registry import get_tokenizer_config
from helm.common.cache import CacheConfig, MongoCacheConfig, SqliteCacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import create_object, inject_object_spec_args
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.proxy.clients.client import Client
from helm.proxy.critique.critique_client import CritiqueClient
from helm.proxy.clients.toxicity_classifier_client import ToxicityClassifierClient
from helm.proxy.retry import NonRetriableException, retry_request
from helm.proxy.tokenizers.tokenizer import Tokenizer
from helm.proxy.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer

from .http_model_client import HTTPModelClient

if TYPE_CHECKING:
    import helm.proxy.clients.huggingface_client


class AuthenticationError(NonRetriableException):
    pass


class AutoClient(Client):
    """Automatically dispatch to the proper `Client` based on the organization.

    The modules for each client are lazily imported when the respective client is created.
    This greatly speeds up the import time of this module, and allows the client modules to
    use optional dependencies."""

    def __init__(self, credentials: Mapping[str, Any], cache_path: str, mongo_uri: str = ""):
        self.credentials = credentials
        self.cache_path = cache_path
        self.mongo_uri = mongo_uri
        self.clients: Dict[str, Client] = {}
        self.tokenizers: Dict[str, Tokenizer] = {}
        # self._huggingface_client is lazily instantiated by get_huggingface_client()
        self._huggingface_client: Optional["helm.proxy.clients.huggingface_client.HuggingFaceClient"] = None
        # self._critique_client is lazily instantiated by get_critique_client()
        self._critique_client: Optional[CritiqueClient] = None
        hlog(f"AutoClient: cache_path = {cache_path}")
        hlog(f"AutoClient: mongo_uri = {mongo_uri}")

    def _build_cache_config(self, organization: str) -> CacheConfig:
        if self.mongo_uri:
            return MongoCacheConfig(self.mongo_uri, collection_name=organization)

        client_cache_path: str = os.path.join(self.cache_path, f"{organization}.sqlite")
        # TODO: Allow setting CacheConfig.follower_cache_path from a command line flag.
        return SqliteCacheConfig(client_cache_path)

    def _get_client(self, model: str) -> Client:
        """Return a client based on the model, creating it if necessary."""
        client: Optional[Client] = self.clients.get(model)

        if client is None:
            organization: str = model.split("/")[0]
            cache_config: CacheConfig = self._build_cache_config(organization)
            tokenizer: Tokenizer = self._get_tokenizer(organization)

            # TODO: Migrate all clients to use model deployments
            model_deployment = get_model_deployment(model)
            if model_deployment:

                def provide_api_key():
                    if "deployments" not in self.credentials:
                        raise AuthenticationError("Could not find key 'deployments' in credentials.conf")
                    deployment_api_keys = self.credentials["deployments"]
                    if model not in deployment_api_keys:
                        raise AuthenticationError(
                            f"Could not find key '{model}' under key 'deployments' in credentials.conf"
                        )
                    return deployment_api_keys[model]

                # Perform dependency injection to fill in remaining arguments.
                # Dependency injection is needed here for these reasons:
                #
                # 1. Different clients have different parameters. Dependency injection provides arguments
                #    that match the parameters of the client.
                # 2. Some arguments, such as the tokenizer, are not static data objects that can be
                #    in the users configuration file. Instead, they have to be constructed dynamically at
                #    runtime.
                # 3. The providers must be lazily-evaluated, because eager evaluation can result in an
                #    exception. For instance, some clients do not require an API key, so trying to fetch
                #    the API key from configuration eagerly will result in an exception because the user
                #    will not have configured an API key.
                client_spec = inject_object_spec_args(
                    model_deployment.client_spec,
                    constant_bindings={"cache_config": cache_config},
                    provider_bindings={"api_key": provide_api_key},
                )
                client = create_object(client_spec)
            elif organization == "neurips":
                client = HTTPModelClient(tokenizer=tokenizer, cache_config=cache_config)
            elif organization == "openai":
                from helm.proxy.clients.openai_client import OpenAIClient

                org_id = self.credentials.get("openaiOrgId", None)
                api_key = self.credentials.get("openaiApiKey", None)
                client = OpenAIClient(
                    tokenizer=tokenizer,
                    cache_config=cache_config,
                    api_key=api_key,
                    org_id=org_id,
                )
            elif organization == "AlephAlpha":
                from helm.proxy.clients.aleph_alpha_client import AlephAlphaClient

                client = AlephAlphaClient(
                    tokenizer=tokenizer,
                    api_key=self.credentials["alephAlphaKey"],
                    cache_config=cache_config,
                )
            elif organization == "ai21":
                from helm.proxy.clients.ai21_client import AI21Client

                client = AI21Client(
                    tokenizer=tokenizer,
                    api_key=self.credentials["ai21ApiKey"],
                    cache_config=cache_config,
                )
            elif organization == "cohere":
                from helm.proxy.clients.cohere_client import CohereClient

                client = CohereClient(
                    tokenizer=tokenizer,
                    api_key=self.credentials["cohereApiKey"],
                    cache_config=cache_config,
                )
            elif organization == "gooseai":
                from helm.proxy.clients.goose_ai_client import GooseAIClient

                org_id = self.credentials.get("gooseaiOrgId", None)
                client = GooseAIClient(
                    tokenizer=tokenizer,
                    api_key=self.credentials["gooseaiApiKey"],
                    cache_config=cache_config,
                    org_id=org_id,
                )
            elif organization == "huggingface":
                from helm.proxy.clients.huggingface_client import HuggingFaceClient

                client = HuggingFaceClient(tokenizer=tokenizer, cache_config=cache_config)
            elif organization == "anthropic":
                from helm.proxy.clients.anthropic_client import AnthropicClient

                client = AnthropicClient(
                    api_key=self.credentials.get("anthropicApiKey", None),
                    tokenizer=tokenizer,
                    cache_config=cache_config,
                )
            elif organization == "microsoft":
                from helm.proxy.clients.microsoft_client import MicrosoftClient

                org_id = self.credentials.get("microsoftOrgId", None)
                lock_file_path: str = os.path.join(self.cache_path, f"{organization}.lock")
                client = MicrosoftClient(
                    api_key=self.credentials.get("microsoftApiKey", None),
                    tokenizer=tokenizer,
                    lock_file_path=lock_file_path,
                    cache_config=cache_config,
                    org_id=org_id,
                )
            elif organization == "google":
                from helm.proxy.clients.google_client import GoogleClient

                client = GoogleClient(
                    tokenizer=tokenizer,
                    cache_config=cache_config,
                )
            elif organization in [
                "together",
                "databricks",
                "eleutherai",
                "lmsys",
                "meta",
                "mistralai",
                "mosaicml",
                "stabilityai",
                "stanford",
                "tiiuae",
            ]:
                from helm.proxy.clients.together_client import TogetherClient

                client = TogetherClient(
                    api_key=self.credentials.get("togetherApiKey", None),
                    tokenizer=tokenizer,
                    cache_config=cache_config,
                )
            elif organization == "simple":
                from helm.proxy.clients.simple_client import SimpleClient

                client = SimpleClient(tokenizer=tokenizer, cache_config=cache_config)
            elif organization == "writer":
                from helm.proxy.clients.palmyra_client import PalmyraClient

                client = PalmyraClient(
                    api_key=self.credentials["writerApiKey"],
                    tokenizer=tokenizer,
                    cache_config=cache_config,
                )
            elif organization == "nvidia":
                from helm.proxy.clients.megatron_client import MegatronClient

                client = MegatronClient(tokenizer=tokenizer, cache_config=cache_config)

            elif organization == "lightningai":
                from helm.proxy.clients.lit_gpt_client import LitGPTClient

                client = LitGPTClient(
                    tokenizer=tokenizer,
                    cache_config=cache_config,
                    checkpoint_dir=Path(os.environ.get("LIT_GPT_CHECKPOINT_DIR", "")),
                    precision=os.environ.get("LIT_GPT_PRECISION", "bf16-true"),
                )
            elif organization == "HuggingFaceM4":
                from helm.proxy.clients.vision_language.idefics_client import IDEFICSClient

                client = IDEFICSClient(tokenizer=tokenizer, cache_config=cache_config)
            else:
                raise ValueError(f"Could not find client for model: {model}")
            self.clients[model] = client
        return client

    def make_request(self, request: Request) -> RequestResult:
        """
        Dispatch based on the name of the model (e.g., openai/davinci).
        Retries if request fails.
        """

        # TODO: need to revisit this because this swallows up any exceptions that are raised.
        @retry_request
        def make_request_with_retry(client: Client, request: Request) -> RequestResult:
            return client.make_request(request)

        client: Client = self._get_client(request.model)

        try:
            return make_request_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = (
                f"Failed to make request to {request.model} after retrying {last_attempt.attempt_number} times"
            )
            hlog(retry_error)

            # Notify our user that we failed to make the request even after retrying.
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def _get_tokenizer(self, tokenizer_name: str) -> Tokenizer:
        # First try to find the tokenizer in the cache
        tokenizer: Optional[Tokenizer] = self.tokenizers.get(tokenizer_name)
        if tokenizer is not None:
            return tokenizer

        # Otherwise, create the tokenizer
        organization: str = tokenizer_name.split("/")[0]
        cache_config: CacheConfig = self._build_cache_config(organization)

        # TODO: Migrate all clients to use tokenizer configs
        tokenizer_config = get_tokenizer_config(tokenizer_name)
        if tokenizer_config:
            tokenizer_spec = inject_object_spec_args(
                tokenizer_config.tokenizer_spec, constant_bindings={"cache_config": cache_config}
            )
            return create_object(tokenizer_spec)
        elif organization in [
            "gooseai",
            "huggingface",
            "microsoft",
            "google",
            "writer",  # Palmyra
            "nvidia",
            "EleutherAI",
            "facebook",
            "meta-llama",
            "hf-internal-testing",
            "mistralai",
            "HuggingFaceM4",
            # Together
            "together",
            "databricks",
            "eleutherai",
            "lmsys",
            "meta",
            "mosaicml",
            "stabilityai",
            "stanford",
            "tiiuae",
            "bigcode",
            "bigscience",
        ]:
            from helm.proxy.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer

            tokenizer = HuggingFaceTokenizer(cache_config=cache_config)
        elif organization == "neurips":
            from helm.proxy.tokenizers.http_model_tokenizer import HTTPModelTokenizer

            tokenizer = HTTPModelTokenizer(cache_config=cache_config)
        elif organization == "openai":
            from helm.proxy.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

            tokenizer = TiktokenTokenizer(cache_config=cache_config)
        elif organization == "AlephAlpha":
            from helm.proxy.tokenizers.aleph_alpha_tokenizer import AlephAlphaTokenizer

            tokenizer = AlephAlphaTokenizer(api_key=self.credentials["alephAlphaKey"], cache_config=cache_config)
        elif organization == "ai21":
            from helm.proxy.tokenizers.ai21_tokenizer import AI21Tokenizer

            tokenizer = AI21Tokenizer(api_key=self.credentials["ai21ApiKey"], cache_config=cache_config)
        elif organization == "cohere":
            from helm.proxy.tokenizers.cohere_tokenizer import CohereTokenizer

            tokenizer = CohereTokenizer(api_key=self.credentials["cohereApiKey"], cache_config=cache_config)
        elif organization == "anthropic":
            from helm.proxy.tokenizers.anthropic_tokenizer import AnthropicTokenizer

            tokenizer = AnthropicTokenizer(cache_config=cache_config)
        elif organization == "simple":
            from helm.proxy.tokenizers.simple_tokenizer import SimpleTokenizer

            tokenizer = SimpleTokenizer()
        elif organization == "lightningai":
            from helm.proxy.tokenizers.lit_gpt_tokenizer import LitGPTTokenizer

            tokenizer = LitGPTTokenizer(
                cache_config=cache_config,
                checkpoint_dir=Path(os.environ.get("LIT_GPT_CHECKPOINT_DIR", "")),
            )
        elif organization == "TsinghuaKEG":
            from helm.proxy.tokenizers.ice_tokenizer import ICETokenizer

            tokenizer = ICETokenizer(cache_config=cache_config)
        elif organization == "Yandex":
            from helm.proxy.tokenizers.yalm_tokenizer import YaLMTokenizer

            tokenizer = YaLMTokenizer(cache_config=cache_config)

        if tokenizer is None:
            raise ValueError(f"Could not find tokenizer for model: {tokenizer_name}")

        # Cache the tokenizer
        self.tokenizers[tokenizer_name] = tokenizer

        return tokenizer

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes based on the name of the tokenizer (e.g., huggingface/gpt2)."""

        def tokenize_with_retry(tokenizer: Tokenizer, request: TokenizationRequest) -> TokenizationRequestResult:
            return tokenizer.tokenize(request)

        tokenizer: Tokenizer = self._get_tokenizer(request.tokenizer)

        try:
            return tokenize_with_retry(tokenizer=tokenizer, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to tokenize after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes based on the the name of the tokenizer (e.g., huggingface/gpt2)."""

        def decode_with_retry(tokenizer: Tokenizer, request: DecodeRequest) -> DecodeRequestResult:
            return tokenizer.decode(request)

        tokenizer: Tokenizer = self._get_tokenizer(request.tokenizer)

        try:
            return decode_with_retry(tokenizer=tokenizer, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to decode after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def get_toxicity_classifier_client(self) -> ToxicityClassifierClient:
        """Get the toxicity classifier client. We currently only support Perspective API."""
        from helm.proxy.clients.perspective_api_client import PerspectiveAPIClient

        cache_config: CacheConfig = self._build_cache_config("perspectiveapi")
        return PerspectiveAPIClient(self.credentials.get("perspectiveApiKey", ""), cache_config)

    def get_critique_client(self) -> CritiqueClient:
        """Get the critique client."""
        if self._critique_client:
            return self._critique_client
        critique_type = self.credentials.get("critiqueType")
        if critique_type == "random":
            from helm.proxy.critique.critique_client import RandomCritiqueClient

            self._critique_client = RandomCritiqueClient()
        elif critique_type == "mturk":
            from helm.proxy.critique.mechanical_turk_critique_client import (
                MechanicalTurkCritiqueClient,
            )

            self._critique_client = MechanicalTurkCritiqueClient()
        elif critique_type == "surgeai":
            from helm.proxy.critique.surge_ai_critique_client import (
                SurgeAICritiqueClient,
            )

            surgeai_credentials = self.credentials.get("surgeaiApiKey")
            if not surgeai_credentials:
                raise ValueError("surgeaiApiKey credentials are required for SurgeAICritiqueClient")
            self._critique_client = SurgeAICritiqueClient(surgeai_credentials, self._build_cache_config("surgeai"))
        elif critique_type == "model":
            from helm.proxy.critique.model_critique_client import ModelCritiqueClient

            model_name: Optional[str] = self.credentials.get("critiqueModelName")
            if model_name is None:
                raise ValueError("critiqueModelName is required for ModelCritiqueClient")
            client: Client = self._get_client(model_name)
            self._critique_client = ModelCritiqueClient(client, model_name)
        elif critique_type == "scale":
            from helm.proxy.critique.scale_critique_client import ScaleCritiqueClient

            scale_credentials = self.credentials.get("scaleApiKey")
            scale_project = self.credentials.get("scaleProject", None)
            if not scale_project:
                raise ValueError("scaleProject is required for ScaleCritiqueClient.")
            if not scale_credentials:
                raise ValueError("scaleApiKey is required for ScaleCritiqueClient")
            self._critique_client = ScaleCritiqueClient(
                scale_credentials, self._build_cache_config("scale"), scale_project
            )
        else:
            raise ValueError(
                "CritiqueClient is not configured; set critiqueType to 'mturk',"
                "'mturk-sandbox', 'surgeai', 'scale' or 'random'"
            )
        return self._critique_client

    def get_huggingface_client(self) -> "helm.proxy.clients.huggingface_client.HuggingFaceClient":
        """Get the Hugging Face client."""
        from helm.proxy.clients.huggingface_client import HuggingFaceClient

        if self._huggingface_client:
            assert isinstance(self._huggingface_client, HuggingFaceClient)
            return self._huggingface_client
        cache_config = self._build_cache_config("huggingface")
        tokenizer = HuggingFaceTokenizer(cache_config)
        self._huggingface_client = HuggingFaceClient(tokenizer=tokenizer, cache_config=cache_config)
        return self._huggingface_client
