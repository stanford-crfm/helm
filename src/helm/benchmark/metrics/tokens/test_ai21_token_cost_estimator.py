from unittest.mock import Mock

from helm.common.request import Request
from helm.benchmark.metrics.tokens.ai21_token_cost_estimator import AI21TokenCostEstimator


class TestAI21TokenCostEstimator:
    def setup_method(self, method):
        self._token_cost_estimator = AI21TokenCostEstimator()

    def test_estimate_tokens(self):
        request = Request(
            model="ai21/jamba-instruct",
            model_deployment="ai21/jamba-instruct",
            prompt="The Center for Research on Foundation Models (CRFM) is "
            "an interdisciplinary initiative born out of the Stanford "
            "Institute for Human-Centered Artificial Intelligence (HAI) "
            "that aims to make fundamental advances in the study, development, "
            "and deployment of foundation models.",
            num_completions=3,
            max_tokens=100,
        )

        # AI21 doesn't count tokens from the prompt.
        # So, just count the maximum number of tokens from completions = 3 * 100
        assert self._token_cost_estimator.estimate_tokens(request, Mock()) == 3 * 100
