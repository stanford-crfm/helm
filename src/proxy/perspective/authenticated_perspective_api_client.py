import threading
from typing import List, Dict, Tuple

from dacite import from_dict
from googleapiclient import discovery
from googleapiclient.errors import BatchError, HttpError
from googleapiclient.http import BatchHttpRequest
from httplib2 import HttpLib2Error

from common.cache import Cache
from common.request import ToxicityAttributes, RequestResult
from proxy.perspective.perspective_api_client import PerspectiveAPIClient


class PerspectiveAPIClientError(Exception):
    pass


class AuthenticatedPerspectiveAPIClient(PerspectiveAPIClient):
    """
    Authenticated Perspective API client.
    """

    # List of supported attributes are found here:
    # https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages
    _DEFAULT_ATTRIBUTES = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "IDENTITY_ATTACK",
        "INSULT",
        "PROFANITY",
        "THREAT",
        "SEXUALLY_EXPLICIT",
        "FLIRTATION",
    ]

    @staticmethod
    def create_perspective_api_request(text: str) -> Dict:
        # TODO: Support english for now. Some of the non-english languages don't support all the default attributes.
        return {
            "comment": {"text": text},
            "requestedAttributes": {
                attribute: {} for attribute in AuthenticatedPerspectiveAPIClient._DEFAULT_ATTRIBUTES
            },
            "languages": ["en"],
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

    def __init__(self, api_key: str, cache_path: str):
        # Google API client
        self._client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        # Cache requests and responses from Perspective API
        self._cache = Cache(cache_path)
        # httplib2 is not thread-safe. Acquire this lock when sending requests
        self._request_lock = threading.RLock()

    def set_toxicity_attributes(self, request_result: RequestResult):
        """
        Send a batch request for all the completions and set the toxicity attributes for each completion.
        """
        text_batch: List[str] = [completion.text for completion in request_result.completions]
        text_to_toxicity_attributes: Dict[str, ToxicityAttributes] = (self.get_toxicity_attributes_batch(text_batch))
        for completion in request_result.completions:
            completion.toxicity_attributes = text_to_toxicity_attributes[completion.text]

    def get_toxicity_attributes_batch(self, text_batch: List[str]) -> Dict[str, ToxicityAttributes]:
        """
        Batch several requests into a single API request to get the toxicity attributes for a batch of text.
        For more information, see https://googleapis.github.io/google-api-python-client/docs/batch.html.
        """

        text_to_response: Dict[str, Dict] = dict()
        text_to_cache_key: Dict[str, Dict] = dict()

        def callback(request_id, response, exception):
            if exception:
                raise PerspectiveAPIClientError(
                    f"Error occurred when making a batch request to Perspective API: {exception}"
                )
            text_to_response[request_id] = response

        # Create a batch request. We will add a request to the batch request for each text string
        batch_request: BatchHttpRequest = self._client.new_batch_http_request()

        # Deduplicate the list of text, since we use the text as the unique identifier when processing the response
        for text in set(text_batch):
            request: Dict = AuthenticatedPerspectiveAPIClient.create_perspective_api_request(text)
            response, cached = self._cache.get(request)

            assert cached
            # If the response is not cached, add the request to the batch request
            if cached:
                text_to_response[text] = response
            else:
                text_to_cache_key[text] = request
                batch_request.add(
                    request=self._client.comments().analyze(body=request), request_id=text, callback=callback,
                )

        try:
            with self._request_lock:
                batch_request.execute()

        except (BatchError, HttpLib2Error) as e:
            raise PerspectiveAPIClientError(f"Error was thrown when making the request to Perspective API: {e}")

        requests_responses_to_cache: List[Tuple[Dict, Dict]] = []
        text_to_toxicity_attributes: Dict[str, ToxicityAttributes] = {}
        for text, response in text_to_response.items():
            text_to_toxicity_attributes[text] = AuthenticatedPerspectiveAPIClient.extract_toxicity_attributes(response)
            if text in text_to_cache_key:
                cache_key: Dict = text_to_cache_key[text]
                requests_responses_to_cache.append((cache_key, response))

        self._cache.bulk_update(requests_responses_to_cache)
        return text_to_toxicity_attributes

    def get_toxicity_attributes(self, text: str) -> ToxicityAttributes:
        """Get the toxicity attributes of a given text."""
        request: Dict = AuthenticatedPerspectiveAPIClient.create_perspective_api_request(text)

        try:

            def do_it():
                with self._request_lock:
                    return self._client.comments().analyze(body=request).execute()

            response, _ = self._cache.get_or_compute(request, do_it)
        except HttpError as e:
            raise PerspectiveAPIClientError(f"Error occurred when making the request to Perspective API: {e}")

        toxicity_attributes: ToxicityAttributes = AuthenticatedPerspectiveAPIClient.extract_toxicity_attributes(
            response
        )
        return toxicity_attributes
