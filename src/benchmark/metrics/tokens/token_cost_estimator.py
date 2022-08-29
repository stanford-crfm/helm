from abc import ABC, abstractmethod

from benchmark.metrics.metric_service import MetricService
from common.request import Request


class TokenCostEstimator(ABC):
    """Estimates token cost given a `Request`."""

    @abstractmethod
    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Given a request, estimate the number of tokens.
        """
        pass
