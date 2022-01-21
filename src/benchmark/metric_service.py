from .executor import Executor
from common.authentication import Authentication
from common.perspective_api_request import PerspectiveAPIRequest
from proxy.remote_service import RemoteService


class MetricService:
    """
    A wrapper around `Executor` that makes only necessary server requests when calculating metrics.
    """

    def __init__(self, executor: Executor):
        self._auth: Authentication = executor.execution_spec.auth
        self._remote_service: RemoteService = executor.remote_service

    def get_toxicity_scores(self, request: PerspectiveAPIRequest):
        return self._remote_service.get_toxicity_scores(self._auth, request)
