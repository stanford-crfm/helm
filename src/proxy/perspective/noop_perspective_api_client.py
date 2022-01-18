from common.request import RequestResult
from proxy.perspective.perspective_api_client import PerspectiveAPIClient


class NoopPerspectiveAPIClient(PerspectiveAPIClient):
    """
    Does nothing.
    """

    def set_toxicity_attributes(self, request_result: RequestResult):
        pass
