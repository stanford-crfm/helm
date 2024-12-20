from helm.benchmark.metrics.metric_service import MetricService
from helm.common.request import Request
from helm.benchmark.metrics.tokens.token_cost_estimator import TokenCostEstimator


class CohereTokenCostEstimator(TokenCostEstimator):
    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Cohere charges by the number of characters in the completion, but first, compute
        the max number of tokens are in the output.
        """
        return request.num_completions * request.max_tokens
