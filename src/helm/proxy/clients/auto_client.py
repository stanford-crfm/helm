import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from retrying import Attempt, RetryError

from helm.benchmark.model_deployment_registry import ModelDeployment, get_model_deployment
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

    def _provide_api_key(self, host_group: str, model: Optional[str] = None) -> Optional[str]:
        api_key_name = host_group + "ApiKey"
        if api_key_name in self.credentials:
            hlog(f"Using host_group api key defined in credentials.conf: {api_key_name}")
            return self.credentials[api_key_name]
        if "deployments" not in self.credentials:
            hlog(
                "WARNING: Could not find key 'deployments' in credentials.conf, "
                f"therefore the API key {api_key_name} should be specified."
            )
            return None
        deployment_api_keys = self.credentials["deployments"]
        if model is None:
            hlog(f"WARNING: Could not find key '{host_group}' in credentials.conf and no model provided")
            return None
        if model not in deployment_api_keys:
            hlog(f"WARNING: Could not find key '{model}' under key 'deployments' in credentials.conf")
            return None
        return deployment_api_keys[model]

    def _get_client(self, model: str) -> Client:
        """Return a client based on the model, creating it if necessary."""
        # First try to find the client in the cache
        client: Optional[Client] = self.clients.get(model)
        if client is not None:
            return client

        # Otherwise, create the client
        host_group: str = model.split("/")[0]
        cache_config: CacheConfig = self._build_cache_config(host_group)

        model_deployment: ModelDeployment = get_model_deployment(model)
        if model_deployment:
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
                provider_bindings={
                    "api_key": lambda: self._provide_api_key(host_group, model),
                    "tokenizer": lambda: self._get_tokenizer(
                        tokenizer_name=model_deployment.tokenizer_name or model_deployment.name,
                        model_deployment_name=model_deployment.name,
                    ),
                    "org_id": lambda: self.credentials.get(host_group + "OrgId", None),  # OpenAI, GooseAI, Microsoft
                    "lock_file_path": lambda: os.path.join(self.cache_path, f"{host_group}.lock"),  # Microsoft
                },
            )
            client = create_object(client_spec)
        else:
            raise ValueError(f"Could not find client for model: {model}")

        # Cache the client
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

    def _get_tokenizer(self, tokenizer_name: str, model_deployment_name: Optional[str] = None) -> Tokenizer:
        # First try to find the tokenizer in the cache
        tokenizer: Optional[Tokenizer] = self.tokenizers.get(tokenizer_name)
        if tokenizer is not None:
            return tokenizer

        # Otherwise, create the tokenizer
        organization: str = tokenizer_name.split("/")[0]
        cache_config: CacheConfig = self._build_cache_config(organization)

        tokenizer_config = get_tokenizer_config(tokenizer_name)
        if tokenizer_config:
            # Callable to get args in the client spec and provide them to the tokenizer.
            def get_client_spec_arg(arg: str) -> Any:
                if model_deployment_name is None:
                    return None
                model_deployment: ModelDeployment = get_model_deployment(model_deployment_name)
                if model_deployment is None:
                    return None
                return model_deployment.client_spec.args.get(arg, None)

            tokenizer_spec = inject_object_spec_args(
                tokenizer_config.tokenizer_spec,
                constant_bindings={"cache_config": cache_config},
                provider_bindings={
                    "api_key": lambda: self._provide_api_key(organization),
                    # TODO: All of these arguments should be auto injected because they are in the client_spec.
                    "checkpoint_dir": lambda: get_client_spec_arg("checkpoint_dir"),  # LitGPT
                    "base_url": lambda: get_client_spec_arg("base_url"),  # HttpModel
                    "do_cache": lambda: get_client_spec_arg("do_cache"),  # HttpModel
                    "pretrained_model_name_or_path": lambda: get_client_spec_arg(
                        "pretrained_model_name_or_path"
                    ),  # HuggingFace
                    "revision": lambda: get_client_spec_arg("revision"),  # HuggingFace
                },
            )
            tokenizer = create_object(tokenizer_spec)  # This returns a Tokenizer but mypy doesn't know that.

        # Cache the tokenizer
        self.tokenizers[tokenizer_name] = tokenizer  # type: ignore

        return tokenizer  # type: ignore

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
