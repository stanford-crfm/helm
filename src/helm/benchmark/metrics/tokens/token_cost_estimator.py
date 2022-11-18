from abc import ABC, abstractmethod

from helm.benchmark.metrics.metric_service import MetricService
from helm.common.request import Request


class TokenCostEstimator(ABC):
    """Estimates token cost given a `Request`."""

    @abstractmethod
    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Given a request, estimate the number of tokens.
        """
        pass
