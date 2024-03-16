from typing import Optional

from helm.common.authentication import Authentication
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.file_upload_request import FileUploadResult, FileUploadRequest
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult
from helm.common.clip_score_request import CLIPScoreRequest, CLIPScoreResult
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.proxy.services.service import Service
from helm.common.cache import Cache


class MetricService(TokenizerService):
    """
    A wrapper around `Service` that makes only necessary server requests when calculating metrics.
    """

    def __init__(self, service: Service, auth: Authentication):
        super().__init__(service, auth)

    def check_nudity(self, request: NudityCheckRequest) -> NudityCheckResult:
        return self._service.check_nudity(self._auth, request)

    def compute_clip_score(self, request: CLIPScoreRequest) -> CLIPScoreResult:
        return self._service.compute_clip_score(self._auth, request)

    def upload(self, request: FileUploadRequest) -> FileUploadResult:
        return self._service.upload(self._auth, request)

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        return self._service.get_toxicity_scores(self._auth, request)

    def make_critique_request(self, request: CritiqueRequest) -> Optional[CritiqueRequestResult]:
        return self._service.make_critique_request(self._auth, request)

    def get_cache(self, shard_name: str) -> Cache:
        return Cache(self._service.get_cache_config(shard_name))
