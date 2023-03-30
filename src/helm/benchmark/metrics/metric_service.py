from typing import Optional

from helm.common.authentication import Authentication
from helm.common.critique_request import CritiqueRequest, CritiqueRequestResult
from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.proxy.services.service import Service


class MetricService(TokenizerService):
    """
    A wrapper around `Service` that makes only necessary server requests when calculating metrics.
    """

    def __init__(self, service: Service, auth: Authentication):
        super().__init__(service, auth)

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        return self._service.get_toxicity_scores(self._auth, request)

    def make_critique_request(self, request: CritiqueRequest) -> Optional[CritiqueRequestResult]:
        return self._service.make_critique_request(self._auth, request)
