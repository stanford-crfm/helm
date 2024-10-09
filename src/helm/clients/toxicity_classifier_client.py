from abc import ABC, abstractmethod

from helm.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult


class ToxicityClassifierClient(ABC):
    """A client that gets toxicity attributes and scores"""

    @abstractmethod
    def get_toxicity_scores(self, request: PerspectiveAPIRequest) -> PerspectiveAPIRequestResult:
        """Get the toxicity attributes and scores for a batch of text."""
        raise NotImplementedError()
