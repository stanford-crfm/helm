from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.common.request import Request
from helm.proxy.token_counters.gooseai_token_counter import GooseAITokenCounter
from .token_cost_estimator import TokenCostEstimator


class GooseAITokenCostEstimator(TokenCostEstimator):
    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Estimate the number of generated tokens for a given request. Formula:

            num_completions * max_tokens

        Add num_tokens(prompt) if `Request.echo_prompt` is True.
        """
        total_estimated_tokens: int = request.num_completions * request.max_tokens
        if request.echo_prompt:
            window_service: WindowService = WindowServiceFactory.get_window_service(request.model, metric_service)
            total_estimated_tokens += window_service.get_num_tokens(request.prompt)
        return GooseAITokenCounter.account_for_base_tokens(total_estimated_tokens)
