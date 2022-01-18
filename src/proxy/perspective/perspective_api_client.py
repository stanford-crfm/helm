from abc import ABC, abstractmethod

from common.request import RequestResult


class PerspectiveAPIClient(ABC):
    """
    Perspective API predicts the perceived impact a comment may have on a conversation by evaluating that comment
    across a range of emotional concepts, called attributes. When you send a request to the API, you’ll request the
    specific attributes you want to receive scores for.

    The API is hosted on Google Cloud Platform.

    Source: https://developers.perspectiveapi.com/s/docs
    """

    @abstractmethod
    def set_toxicity_attributes(self, request_result: RequestResult):
        """
        Get and set the toxicity attributes for the completions of `request_result`.
        """
        pass
