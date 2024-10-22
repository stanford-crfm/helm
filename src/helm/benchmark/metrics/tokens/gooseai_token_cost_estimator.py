from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.common.request import Request
from helm.benchmark.metrics.tokens.token_cost_estimator import TokenCostEstimator


class GooseAITokenCostEstimator(TokenCostEstimator):
    # From https://goose.ai/pricing: "the base price includes your first 25 tokens
    # generated, and you can scale beyond that on a per-token basis."
    BASE_PRICE_TOKENS: int = 25

    @staticmethod
    def account_for_base_tokens(num_tokens: int):
        """Subtracts the number of tokens included in the base price."""
        return max(num_tokens - GooseAITokenCostEstimator.BASE_PRICE_TOKENS, 0)

    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Estimate the number of generated tokens for a given request. Formula:

            num_completions * max_tokens

        Add num_tokens(prompt) if `Request.echo_prompt` is True.
        """
        total_estimated_tokens: int = request.num_completions * request.max_tokens
        if request.echo_prompt:
            window_service: WindowService = WindowServiceFactory.get_window_service(
                request.model_deployment, metric_service
            )
            total_estimated_tokens += window_service.get_num_tokens(request.prompt)
        return GooseAITokenCostEstimator.account_for_base_tokens(total_estimated_tokens)
