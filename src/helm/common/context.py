from abc import ABC, abstractmethod

from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.clip_score_request import CLIPScoreRequest, CLIPScoreResult
from helm.common.file_upload_request import FileUploadResult, FileUploadRequest
from helm.common.nudity_check_request import NudityCheckRequest, NudityCheckResult
from helm.common.perspective_api_request import PerspectiveAPIRequestResult, PerspectiveAPIRequest
from helm.common.moderations_api_request import ModerationAPIRequest, ModerationAPIRequestResult
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.common.request import Request, RequestResult
from helm.proxy.query import Query, QueryResult
from helm.common.cache import CacheConfig
from helm.proxy.services.service import GeneralInfo


class Context(ABC):
    @abstractmethod
    def get_general_info(self) -> GeneralInfo:
        """Get general info."""
        pass

    @abstractmethod
    def expand_query(self, query: Query) -> QueryResult:
        """Turn the `query` into requests."""
        pass

    @abstractmethod
    def make_request(self, request: Request) -> RequestResult:
        """Actually make a request to an API."""
        pass

    @abstractmethod
    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenize via an API."""
        pass

    @abstractmethod
    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes to text."""
        pass

    @abstractmethod
    def upload(self, request: FileUploadRequest) -> FileUploadResult:
        """Uploads a file to external storage."""
        pass

    @abstractmethod
    def check_nudity(self, request: NudityCheckRequest) -> NudityCheckResult:
        """Check for nudity for a batch of images."""
        pass

    @abstractmethod
    def compute_clip_score(self, request: CLIPScoreRequest) -> CLIPScoreResult:
        """Computes CLIPScore for a given caption and image."""
        pass

    @abstractmethod
    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        """Get toxicity scores for a batch of text."""
        pass

    @abstractmethod
    def get_moderation_results(self, request: ModerationAPIRequest) -> ModerationAPIRequestResult:
        """Get OpenAI's moderation results for some text."""
        pass

    @abstractmethod
    def make_critique_request(self, request: CritiqueRequest) -> CritiqueRequestResult:
        """Get responses to a critique request."""
        pass

    @abstractmethod
    def get_cache_config(self, shard_name: str) -> CacheConfig:
        """Returns a CacheConfig"""
        pass
