from common.authentication import Authentication
from common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from proxy.service import Service


class MetricService:
    """
    A wrapper around `Service` that makes only necessary server requests when calculating metrics.
    """

    def __init__(self, service: Service, auth: Authentication):
        self._service: Service = service
        self._auth: Authentication = auth

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        return self._service.get_toxicity_scores(self._auth, request)
