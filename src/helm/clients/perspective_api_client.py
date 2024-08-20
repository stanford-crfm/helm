# mypy: check_untyped_defs = False
import threading
from dataclasses import asdict
from typing import Any, List, Dict, Optional

from dacite import from_dict

from helm.clients.toxicity_classifier_client import ToxicityClassifierClient
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.proxy.retry import NonRetriableException
from helm.common.cache import Cache, CacheConfig
from helm.common.perspective_api_request import ToxicityAttributes, PerspectiveAPIRequest, PerspectiveAPIRequestResult

try:
    from googleapiclient import discovery
    from googleapiclient.errors import BatchError, HttpError
    from googleapiclient.http import BatchHttpRequest
    from httplib2 import HttpLib2Error
    from google.auth.exceptions import DefaultCredentialsError
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["metrics"])


class PerspectiveAPIClientCredentialsError(NonRetriableException):
    pass


class PerspectiveAPIClient(ToxicityClassifierClient):
    """
    Perspective API predicts the perceived impact a comment may have on a conversation by evaluating that comment
    across a range of emotional concepts, called attributes. When you send a request to the API, you’ll request the
    specific attributes you want to receive scores for.

    The API is hosted on Google Cloud Platform.

    Source: https://developers.perspectiveapi.com/s/docs
    """

    ORGANIZATION = "perspectiveapi"

    # Maximum allowed text length by Perspective API
    MAX_TEXT_LENGTH: int = 20_480

    @staticmethod
    def create_request_body(text: str, attributes: List[str], languages: List[str]) -> Dict:
        """Create an API request for a given text."""
        return {
            "comment": {"text": text},
            "requestedAttributes": {attribute: {} for attribute in attributes},
            "languages": languages,
            "spanAnnotations": True,
        }

    @staticmethod
    def extract_toxicity_attributes(response: Dict) -> ToxicityAttributes:
        """Given a response from PerspectiveAPI, return `ToxicityScores`."""
        all_scores = {
            f"{attribute.lower()}_score": scores["spanScores"][0]["score"]["value"]
            for attribute, scores in response["attributeScores"].items()
        }
        return from_dict(data_class=ToxicityAttributes, data=all_scores)

    def __init__(self, api_key: str, cache_config: CacheConfig):
        # API key obtained from GCP that works with PerspectiveAPI
        self.api_key = api_key

        # Cache requests and responses from Perspective API
        self.cache = Cache(cache_config)

        # httplib2 is not thread-safe. Acquire this lock when sending requests to PerspectiveAPI
        self._client_lock: threading.Lock = threading.Lock()

        # Google Perspective API client.
        # The _client_lock must be held when creating or using the client.
        self._client: Optional[discovery.Resource] = None

    def _create_client(self) -> discovery.Resource:
        """Initialize the client."""
        if not self.api_key:
            raise PerspectiveAPIClientCredentialsError("API key was not set in credentials.conf")
        try:
            return discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=self.api_key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
        except DefaultCredentialsError as e:
            raise PerspectiveAPIClientCredentialsError(
                f"Credentials error when creating Perspective API client: {e}"
            ) from e

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        """
        Batch several requests into a single API request and get the toxicity attributes and scores.
        For more information, see https://googleapis.github.io/google-api-python-client/docs/batch.html.
        """
        try:

            def do_it() -> Dict[str, Any]:
                text_to_response: Dict[str, Dict] = dict()

                def callback(request_id: str, response: Dict, error: HttpError):
                    if error:
                        raise error
                    text_to_response[request_id] = response

                with self._client_lock:
                    if not self._client:
                        self._client = self._create_client()

                # Create a batch request. We will add a request to the batch request for each text string
                batch_request: BatchHttpRequest = self._client.new_batch_http_request()

                # Add individual request to the batch request. Deduplicate since we use the text as request keys.
                for text in set(request.text_batch):
                    batch_request.add(
                        request=self._client.comments().analyze(
                            body=PerspectiveAPIClient.create_request_body(
                                text[: PerspectiveAPIClient.MAX_TEXT_LENGTH], request.attributes, request.languages
                            )
                        ),
                        request_id=text,
                        callback=callback,
                    )

                with self._client_lock:
                    batch_request.execute()
                return text_to_response

            batch_response, cached = self.cache.get(asdict(request), do_it)

        except (BatchError, HttpLib2Error, HttpError) as e:
            return PerspectiveAPIRequestResult(
                success=False,
                cached=False,
                error=f"Error was thrown when making a request to Perspective API: {e}",
            )

        return PerspectiveAPIRequestResult(
            success=True,
            cached=cached,
            text_to_toxicity_attributes={
                text: PerspectiveAPIClient.extract_toxicity_attributes(response)
                for text, response in batch_response.items()
            },
        )
