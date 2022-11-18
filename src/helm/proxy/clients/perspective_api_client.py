import threading
from dataclasses import asdict
from typing import List, Dict, Optional

from dacite import from_dict
from googleapiclient import discovery
from googleapiclient.errors import BatchError, HttpError
from googleapiclient.http import BatchHttpRequest
from httplib2 import HttpLib2Error

from helm.common.cache import Cache, CacheConfig
from helm.common.perspective_api_request import ToxicityAttributes, PerspectiveAPIRequest, PerspectiveAPIRequestResult


class PerspectiveAPIClientError(Exception):
    pass


class PerspectiveAPIClient:
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

        # Google API client
        self.client: Optional[discovery.Resource] = None

        # Cache requests and responses from Perspective API
        self.cache = Cache(cache_config)

        # httplib2 is not thread-safe. Acquire this lock when sending requests to PerspectiveAPI
        self.request_lock: Optional[threading.RLock] = threading.RLock()
        # self.request_lock = None  # TODO: temporary hack to get multiprocessing to work for now

    def _get_or_initialize_client(self) -> discovery.Resource:
        if not self.client:
            try:
                self.client = discovery.build(
                    "commentanalyzer",
                    "v1alpha1",
                    developerKey=self.api_key,
                    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                    static_discovery=False,
                )
            except (HttpError, KeyError) as e:
                raise PerspectiveAPIClientError(
                    f"An error occurred while authenticating and instantiating a client: {e}"
                )
        return self.client

    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        """
        Batch several requests into a single API request and get the toxicity attributes and scores.
        For more information, see https://googleapis.github.io/google-api-python-client/docs/batch.html.
        """
        client = self._get_or_initialize_client()

        try:

            def do_it():
                text_to_response: Dict[str, Dict] = dict()

                def callback(request_id: str, response: Dict, error: HttpError):
                    if error:
                        raise error
                    text_to_response[request_id] = response

                # Create a batch request. We will add a request to the batch request for each text string
                batch_request: BatchHttpRequest = client.new_batch_http_request()

                # Add individual request to the batch request. Deduplicate since we use the text as request keys.
                for text in set(request.text_batch):
                    batch_request.add(
                        request=client.comments().analyze(
                            body=PerspectiveAPIClient.create_request_body(
                                text[: PerspectiveAPIClient.MAX_TEXT_LENGTH], request.attributes, request.languages
                            )
                        ),
                        request_id=text,
                        callback=callback,
                    )

                with self.request_lock:
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
