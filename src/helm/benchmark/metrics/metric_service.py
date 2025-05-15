from typing import Optional

from helm.common.context import Context
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.file_upload_request import FileUploadResult, FileUploadRequest
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult
from helm.common.clip_score_request import CLIPScoreRequest, CLIPScoreResult
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.common.cache import Cache


# TODO: Rename this to TokenizerContext
class MetricService(TokenizerService):
    """
    A wrapper around `Context` that makes only necessary server requests when calculating metrics.
    """

    def __init__(self, context: Context):
        super().__init__(context)

    def check_nudity(self, request: NudityCheckRequest) -> NudityCheckResult:
        return self._context.check_nudity(request)

    def compute_clip_score(self, request: CLIPScoreRequest) -> CLIPScoreResult:
        return self._context.compute_clip_score(request)

    def upload(self, request: FileUploadRequest) -> FileUploadResult:
        return self._context.upload(request)

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        return self._context.get_toxicity_scores(request)

    def make_critique_request(self, request: CritiqueRequest) -> Optional[CritiqueRequestResult]:
        return self._context.make_critique_request(request)

    def get_cache(self, shard_name: str) -> Cache:
        return Cache(self._context.get_cache_config(shard_name))
