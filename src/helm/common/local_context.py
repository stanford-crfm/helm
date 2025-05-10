import dataclasses
import os
from typing import Optional

from helm.common.context import Context
from helm.common.cache import CacheConfig
from helm.common.cache_backend_config import CacheBackendConfig, BlackHoleCacheBackendConfig
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.moderations_api_request import ModerationAPIRequest, ModerationAPIRequestResult
from helm.common.clip_score_request import CLIPScoreRequest, CLIPScoreResult
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult
from helm.common.file_upload_request import FileUploadRequest, FileUploadResult
from helm.common.general import ensure_directory_exists, parse_hocon, get_credentials
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.common.request import Request, RequestResult
from helm.clients.auto_client import AutoClient
from helm.clients.moderation_api_client import ModerationAPIClient
from helm.clients.image_generation.nudity_check_client import NudityCheckClient
from helm.clients.gcs_client import GCSClient
from helm.clients.clip_score_client import CLIPScoreClient
from helm.clients.toxicity_classifier_client import ToxicityClassifierClient
from helm.proxy.example_queries import example_queries
from helm.benchmark.model_metadata_registry import ALL_MODELS_METADATA
from helm.proxy.query import Query, QueryResult
from helm.proxy.retry import retry_request
from helm.tokenizers.auto_tokenizer import AutoTokenizer
from helm.proxy.services.service import (
    CACHE_DIR,
    GeneralInfo,
    VERSION,
    expand_environments,
    synthesize_request,
)


class LocalContext(Context):
    """
    Main class that supports various functionality for the server.
    """

    def __init__(
        self,
        base_path: str = "prod_env",
        cache_backend_config: CacheBackendConfig = BlackHoleCacheBackendConfig(),
    ):
        ensure_directory_exists(base_path)
        client_file_storage_path = os.path.join(base_path, CACHE_DIR)
        ensure_directory_exists(client_file_storage_path)

        credentials = get_credentials(base_path)

        self.cache_backend_config = cache_backend_config
        self.client = AutoClient(credentials, client_file_storage_path, cache_backend_config)
        self.tokenizer = AutoTokenizer(credentials, cache_backend_config)

        # Lazily instantiate the following clients
        self.moderation_api_client: Optional[ModerationAPIClient] = None
        self.toxicity_classifier_client: Optional[ToxicityClassifierClient] = None
        self.perspective_api_client: Optional[ToxicityClassifierClient] = None
        self.nudity_check_client: Optional[NudityCheckClient] = None
        self.clip_score_client: Optional[CLIPScoreClient] = None
        self.gcs_client: Optional[GCSClient] = None

    def get_general_info(self) -> GeneralInfo:
        # Can't send release_dates in ModelMetadata bacause dates cannot be round-tripped to and from JSON easily.
        # TODO(#2158): Either fix this or delete get_general_info.
        all_models = [dataclasses.replace(model_metadata, release_date=None) for model_metadata in ALL_MODELS_METADATA]
        return GeneralInfo(version=VERSION, example_queries=example_queries, all_models=all_models)

    def expand_query(self, query: Query) -> QueryResult:
        """Turn the `query` into requests."""
        prompt = query.prompt
        settings = query.settings
        environments = parse_hocon(query.environments)
        requests = []
        for environment in expand_environments(environments):
            request = synthesize_request(prompt, settings, environment)
            requests.append(request)
        return QueryResult(requests=requests)

    def make_request(self, request: Request) -> RequestResult:
        """Actually make a request to an API."""
        return self.client.make_request(request)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        return self.tokenizer.tokenize(request)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        return self.tokenizer.decode(request)

    def upload(self, request: FileUploadRequest) -> FileUploadResult:
        if not self.gcs_client:
            self.gcs_client = self.client.get_gcs_client()

        assert self.gcs_client
        return self.gcs_client.upload(request)

    def check_nudity(self, request: NudityCheckRequest) -> NudityCheckResult:
        if not self.nudity_check_client:
            self.nudity_check_client = self.client.get_nudity_check_client()

        assert self.nudity_check_client
        return self.nudity_check_client.check_nudity(request)

    def compute_clip_score(self, request: CLIPScoreRequest) -> CLIPScoreResult:
        if not self.clip_score_client:
            self.clip_score_client = self.client.get_clip_score_client()

        assert self.clip_score_client
        return self.clip_score_client.compute_score(request)

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        @retry_request
        def get_toxicity_scores_with_retry(request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
            if not self.toxicity_classifier_client:
                self.toxicity_classifier_client = self.client.get_toxicity_classifier_client()
            return self.toxicity_classifier_client.get_toxicity_scores(request)

        return get_toxicity_scores_with_retry(request)

    def get_moderation_results(self, request: ModerationAPIRequest) -> ModerationAPIRequestResult:
        @retry_request
        def get_moderation_results_with_retry(request: ModerationAPIRequest) -> ModerationAPIRequestResult:
            if not self.moderation_api_client:
                self.moderation_api_client = self.client.get_moderation_api_client()
            return self.moderation_api_client.get_moderation_results(request)

        return get_moderation_results_with_retry(request)

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        return self.client.get_critique_client().make_critique_request(request)

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        return self.cache_backend_config.get_cache_config(shard_name)
