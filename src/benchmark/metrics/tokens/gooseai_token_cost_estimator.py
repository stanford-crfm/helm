from benchmark.metric_service import MetricService
from common.request import Request
from proxy.tokenizer.tokenizer import Tokenizer
from proxy.tokenizer.tokenizer_factory import TokenizerFactory
from proxy.tokenizer.gooseai_token_counter import GooseAITokenCounter
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
            tokenizer: Tokenizer = TokenizerFactory.get_tokenizer("gooseai", metric_service)
            total_estimated_tokens += tokenizer.tokenize_and_count(request.prompt)
        return GooseAITokenCounter.account_for_base_tokens(total_estimated_tokens)
