from helm.benchmark.metrics.metric_service import MetricService
from helm.common.request import Request
from helm.benchmark.metrics.tokens.token_cost_estimator import TokenCostEstimator


class FreeTokenCostEstimator(TokenCostEstimator):
    """For when we don't care about keeping track of the number of tokens."""

    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """No need to estimate tokens, since it's free. Return 0."""
        return 0
