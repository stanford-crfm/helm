from typing import Dict

from helm.benchmark.metrics.metric_service import MetricService
from helm.common.request import Request
from helm.benchmark.metrics.tokens.ai21_token_cost_estimator import AI21TokenCostEstimator
from helm.benchmark.metrics.tokens.cohere_token_cost_estimator import CohereTokenCostEstimator
from helm.benchmark.metrics.tokens.free_token_cost_estimator import FreeTokenCostEstimator
from helm.benchmark.metrics.tokens.gooseai_token_cost_estimator import GooseAITokenCostEstimator
from helm.benchmark.metrics.tokens.openai_token_cost_estimator import OpenAITokenCostEstimator
from helm.benchmark.metrics.tokens.token_cost_estimator import TokenCostEstimator


class AutoTokenCostEstimator(TokenCostEstimator):
    """Automatically count tokens based on the organization."""

    def __init__(self):
        self._token_cost_estimators: Dict[str, TokenCostEstimator] = {}

    def _get_estimator(self, organization: str) -> TokenCostEstimator:
        """Return a `TokenCostEstimator` based on the organization."""
        token_cost_estimator = self._token_cost_estimators.get(organization)

        if token_cost_estimator is None:
            if organization == "openai":
                token_cost_estimator = OpenAITokenCostEstimator()
            elif organization == "ai21":
                token_cost_estimator = AI21TokenCostEstimator()
            elif organization == "cohere":
                token_cost_estimator = CohereTokenCostEstimator()
            elif organization == "gooseai":
                token_cost_estimator = GooseAITokenCostEstimator()
            else:
                token_cost_estimator = FreeTokenCostEstimator()
            self._token_cost_estimators[organization] = token_cost_estimator

        return token_cost_estimator

    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Estimate the number of tokens for a given request based on the organization.
        """
        token_cost_estimator: TokenCostEstimator = self._get_estimator(request.model_host)
        return token_cost_estimator.estimate_tokens(request, metric_service)
