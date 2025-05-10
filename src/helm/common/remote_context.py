from helm.common.context import Context
from helm.common.cache import CacheConfig
from helm.common.authentication import Authentication
from helm.common.moderations_api_request import ModerationAPIRequest, ModerationAPIRequestResult
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult
from helm.common.file_upload_request import FileUploadRequest, FileUploadResult
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.common.clip_score_request import CLIPScoreRequest, CLIPScoreResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequestResult,
    DecodeRequest,
)
from helm.common.request import Request, RequestResult
from helm.proxy.query import Query, QueryResult
from helm.proxy.services.remote_service import RemoteService
from helm.proxy.services.service import GeneralInfo, Service


class RemoteContext(Context):
    def __init__(self, base_url: str, auth: Authentication):
        self.service: Service = RemoteService(base_url)
        self.auth = auth

    def get_general_info(self) -> GeneralInfo:
        return self.service.get_general_info()

    def expand_query(self, query: Query) -> QueryResult:
        return self.service.expand_query(query)

    def make_request(self, request: Request) -> RequestResult:
        return self.service.make_request(self.auth, request)

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        return self.service.tokenize(self.auth, request)

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        return self.service.decode(self.auth, request)

    def upload(self, request: FileUploadRequest) -> FileUploadResult:
        return self.service.upload(self.auth, request)

    def check_nudity(self, request: NudityCheckRequest) -> NudityCheckResult:
        return self.service.check_nudity(self.auth, request)

    def compute_clip_score(self, request: CLIPScoreRequest) -> CLIPScoreResult:
        return self.service.compute_clip_score(self.auth, request)

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        return self.service.get_toxicity_scores(self.auth, request)

    def get_moderation_results(self, request: ModerationAPIRequest) -> ModerationAPIRequestResult:
        return self.service.get_moderation_results(self.auth, request)

    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        return self.service.make_critique_request(self.auth, request)

    def get_cache_config(self, shard_name: str) -> CacheConfig:
        return self.service.get_cache_config(shard_name)
