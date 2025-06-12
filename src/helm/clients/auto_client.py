from dataclasses import replace
import os
from typing import Any, Dict, Mapping, Optional

from retrying import Attempt, RetryError

from helm.benchmark.model_deployment_registry import ModelDeployment, get_model_deployment
from helm.benchmark.tokenizer_config_registry import get_tokenizer_config
from helm.common.file_caches.file_cache import FileCache
from helm.common.file_caches.local_file_cache import LocalFileCache
from helm.common.credentials_utils import provide_api_key
from helm.common.cache_backend_config import CacheBackendConfig, CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.object_spec import create_object, inject_object_spec_args
from helm.common.request import Request, RequestResult
from helm.clients.client import Client
from helm.clients.moderation_api_client import ModerationAPIClient
from helm.proxy.critique.critique_client import CritiqueClient
from helm.clients.toxicity_classifier_client import ToxicityClassifierClient
from helm.proxy.retry import NonRetriableException, retry_request
from helm.tokenizers.auto_tokenizer import AutoTokenizer


class AuthenticationError(NonRetriableException):
    pass


class AutoClient(Client):
    """Automatically dispatch to the proper `Client` based on the model deployment name."""

    def __init__(
        self, credentials: Mapping[str, Any], file_storage_path: str, cache_backend_config: CacheBackendConfig
    ):
        self._auto_tokenizer = AutoTokenizer(credentials, cache_backend_config)
        self.credentials = credentials
        self.file_storage_path = file_storage_path
        self.cache_backend_config = cache_backend_config
        self.clients: Dict[str, Client] = {}
        self._critique_client: Optional[CritiqueClient] = None
        hlog(f"AutoClient: file_storage_path = {file_storage_path}")
        hlog(f"AutoClient: cache_backend_config = {cache_backend_config}")

    def _get_client(self, model_deployment_name: str) -> Client:
        """Return a client based on the model, creating it if necessary."""
        # First try to find the client in the cache
        client: Optional[Client] = self.clients.get(model_deployment_name)
        if client is not None:
            return client

        # Otherwise, create the client
        model_deployment: ModelDeployment = get_model_deployment(model_deployment_name)
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

            # Prepare a cache
            host_organization: str = model_deployment.host_organization
            cache_config: CacheConfig = self.cache_backend_config.get_cache_config(host_organization)

            client_spec = inject_object_spec_args(
                model_deployment.client_spec,
                constant_bindings={
                    "cache_config": cache_config,
                    "model_name": model_deployment.model_name,
                    "tokenizer_name": model_deployment.tokenizer_name,
                },
                provider_bindings={
                    "api_key": lambda: provide_api_key(self.credentials, host_organization, model_deployment_name),
                    "tokenizer": lambda: self._auto_tokenizer._get_tokenizer(
                        tokenizer_name=model_deployment.tokenizer_name or model_deployment.name
                    ),
                    "org_id": lambda: self.credentials.get(
                        host_organization + "OrgId", None
                    ),  # OpenAI, GooseAI, Microsoft
                    "base_url": lambda: self.credentials.get(host_organization + "BaseUrl", None),
                    "moderation_api_client": lambda: self.get_moderation_api_client(),  # OpenAI DALL-E
                    "lock_file_path": lambda: os.path.join(
                        self.file_storage_path, f"{host_organization}.lock"
                    ),  # Microsoft
                    "project_id": lambda: self.credentials.get(host_organization + "ProjectId", None),  # VertexAI
                    "location": lambda: self.credentials.get(host_organization + "Location", None),  # VertexAI
                    "hf_auth_token": lambda: self.credentials.get("huggingfaceAuthToken", None),  # HuggingFace
                    "file_cache": lambda: self._get_file_cache(host_organization),  # Text-to-image models
                    "endpoint": lambda: self.credentials.get(host_organization + "Endpoint", None),  # Palmyra
                    "end_of_text_token": lambda: self._get_end_of_text_token(
                        tokenizer_name=model_deployment.tokenizer_name or model_deployment.name
                    ),
                },
            )
            client = create_object(client_spec)
        else:
            raise ValueError(f"Could not find client for model deployment: {model_deployment_name}")

        # Cache the client
        self.clients[model_deployment_name] = client

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

        client: Client = self._get_client(request.model_deployment)

        try:
            return make_request_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = (
                f"Failed to make request to {request.model_deployment} after retrying "
                f"{last_attempt.attempt_number} times"
            )
            hlog(retry_error)

            # Notify our user that we failed to make the request even after retrying.
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def get_gcs_client(self):
        from helm.clients.gcs_client import GCSClient

        bucket_name: str = self.credentials["gcsBucketName"]
        cache_config: CacheConfig = self.cache_backend_config.get_cache_config("gcs")
        return GCSClient(bucket_name, cache_config)

    def get_nudity_check_client(self):
        from helm.clients.image_generation.nudity_check_client import NudityCheckClient

        cache_config: CacheConfig = self.cache_backend_config.get_cache_config("nudity")
        return NudityCheckClient(cache_config)

    def get_clip_score_client(self):
        from helm.clients.clip_score_client import CLIPScoreClient

        cache_config: CacheConfig = self.cache_backend_config.get_cache_config("clip_score")
        return CLIPScoreClient(cache_config)

    def get_toxicity_classifier_client(self) -> ToxicityClassifierClient:
        """Get the toxicity classifier client. We currently only support Perspective API."""
        from helm.clients.perspective_api_client import PerspectiveAPIClient

        cache_config: CacheConfig = self.cache_backend_config.get_cache_config("perspectiveapi")
        return PerspectiveAPIClient(self.credentials.get("perspectiveApiKey", ""), cache_config)

    def get_moderation_api_client(self) -> ModerationAPIClient:
        """Get the ModerationAPI client."""
        cache_config: CacheConfig = self.cache_backend_config.get_cache_config("ModerationAPI")
        return ModerationAPIClient(self.credentials.get("openaiApiKey", ""), cache_config)

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
            self._critique_client = SurgeAICritiqueClient(
                surgeai_credentials, self.cache_backend_config.get_cache_config("surgeai")
            )
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
                scale_credentials, self.cache_backend_config.get_cache_config("scale"), scale_project
            )
        else:
            raise ValueError(
                "CritiqueClient is not configured; set critiqueType to 'mturk',"
                "'mturk-sandbox', 'surgeai', 'scale' or 'random'"
            )
        return self._critique_client

    def _get_file_cache(self, host_organization: str) -> FileCache:
        # Initialize `FileCache` for text-to-image model APIs
        local_file_cache_path: str = os.path.join(self.file_storage_path, "output", host_organization)
        return LocalFileCache(local_file_cache_path, file_extension="png")

    def _get_end_of_text_token(self, tokenizer_name: str) -> Optional[str]:
        tokenizer_config = get_tokenizer_config(tokenizer_name)
        if tokenizer_config is None:
            raise ValueError(f"Could not find tokenizer_config for tokenizer {tokenizer_name}")
        return tokenizer_config.end_of_text_token
