from typing import Dict

from benchmark.metric_service import MetricService
from common.request import Request
from .ai21_token_cost_estimator import AI21TokenCostEstimator
from .free_token_cost_estimator import FreeTokenCostEstimator
from .gooseai_token_cost_estimator import GooseAITokenCostEstimator
from .openai_token_cost_estimator import OpenAITokenCostEstimator
from .token_cost_estimator import TokenCostEstimator


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
        token_cost_estimator: TokenCostEstimator = self._get_estimator(request.model_organization)
        return token_cost_estimator.estimate_tokens(request, metric_service)
