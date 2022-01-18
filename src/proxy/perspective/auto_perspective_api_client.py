import os
from typing import Dict, Optional

from googleapiclient.errors import HttpError
from proxy.perspective.noop_perspective_api_client import NoopPerspectiveAPIClient
from proxy.perspective.perspective_api_client import PerspectiveAPIClient
from proxy.perspective.authenticated_perspective_api_client import AuthenticatedPerspectiveAPIClient

from common.hierarchical_logger import hlog
from common.request import RequestResult


class AutoPerspectiveAPIClient(PerspectiveAPIClient):
    """
    Use an authenticated PerspectiveAPIClient to calculate the toxicity scores when the credentials are valid.
    Otherwise, do nothing.
    """

    def __init__(self, credentials: Dict[str, str], cache_path: str):
        self.credentials = credentials
        self.client_cache_path = os.path.join(cache_path, "perspectiveapi.sqlite")
        self.client: Optional[PerspectiveAPIClient] = None

    def set_toxicity_attributes(self, request_result: RequestResult):
        """
        Get and set the toxicity attributes for the completions of `request_result`, if authenticated.
        """
        if not self.client:
            try:
                self.client = AuthenticatedPerspectiveAPIClient(
                    api_key=self.credentials["perspectiveApiKey"], cache_path=self.client_cache_path,
                )
            except HttpError as e:
                hlog(f"Disabling Perspective API. An error occurred while authenticating: {e}")
                self.client = NoopPerspectiveAPIClient()

        self.client.set_toxicity_attributes(request_result)
