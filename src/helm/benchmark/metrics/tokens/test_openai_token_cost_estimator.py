# mypy: check_untyped_defs = False
from unittest.mock import MagicMock

from transformers import GPT2TokenizerFast

from helm.benchmark.metrics.metric_service import MetricService
from helm.common.authentication import Authentication
from helm.common.request import Request
from helm.common.tokenization_request import TokenizationRequestResult, TokenizationToken
from helm.proxy.services.remote_service import RemoteService
from helm.benchmark.metrics.tokens.openai_token_cost_estimator import OpenAITokenCostEstimator


class TestOpenAITokenCostEstimator:
    # The following prompt has 51 tokens according to the GPT-2 tokenizer
    TEST_PROMPT: str = (
        "The Center for Research on Foundation Models (CRFM) is "
        "an interdisciplinary initiative born out of the Stanford "
        "Institute for Human-Centered Artificial Intelligence (HAI) "
        "that aims to make fundamental advances in the study, development, "
        "and deployment of foundation models."
    )

    def setup_method(self, method):
        self._token_cost_estimator = OpenAITokenCostEstimator()
        self._mock_metric_service = MetricService(RemoteService("DUMMY_URL"), Authentication(api_key="test"))
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenization_request_result = TokenizationRequestResult(
            success=True,
            cached=False,
            text=TestOpenAITokenCostEstimator.TEST_PROMPT,
            tokens=[
                TokenizationToken(value) for value in gpt2_tokenizer.tokenize(TestOpenAITokenCostEstimator.TEST_PROMPT)
            ],
        )
        self._mock_metric_service.tokenize = MagicMock(return_value=tokenization_request_result)

    def test_estimate_tokens(self):
        request = Request(
            model="openai/davinci-002",
            model_deployment="openai/davinci-002",
            prompt=TestOpenAITokenCostEstimator.TEST_PROMPT,
            num_completions=3,
            max_tokens=100,
        )

        # Prompt + max number of tokens from completions = 51 + 3 * 100
        assert self._token_cost_estimator.estimate_tokens(request, self._mock_metric_service) == 51 + 3 * 100

    def test_estimate_tokens_with_echo_prompt(self):
        request = Request(
            model="openai/davinci-002",
            model_deployment="openai/davinci-002",
            prompt=TestOpenAITokenCostEstimator.TEST_PROMPT,
            echo_prompt=True,
            num_completions=1,
            max_tokens=0,
        )

        # Prompt + Prompt (echoed) = 51 + 51
        assert self._token_cost_estimator.estimate_tokens(request, self._mock_metric_service) == 51 + 51
