from helm.benchmark.metrics.metric_service import MetricService
from helm.common.request import Request
from helm.benchmark.metrics.tokens.token_cost_estimator import TokenCostEstimator


class AI21TokenCostEstimator(TokenCostEstimator):
    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Estimate the number of tokens given a request. We do not need to account for the number
        of tokens in the prompt itself (https://studio.ai21.com/docs/calculating-usage).

        Therefore, estimate using the following formula:

            num_completions * max_tokens
        """
        return request.num_completions * request.max_tokens
