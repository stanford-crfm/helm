from benchmark.metric_service import MetricService
from common.request import Request
from proxy.tokenizer.tokenizer_factory import TokenizerFactory
from proxy.tokenizer.tokenizer import Tokenizer
from .token_cost_estimator import TokenCostEstimator


class OpenAITokenCostEstimator(TokenCostEstimator):
    def estimate_tokens(self, request: Request, metric_service: MetricService) -> int:
        """
        Estimate the number of tokens for a given request. Include the tokens in the prompt
        when estimating number of tokens. Formula:

            num_tokens(prompt) + num_completions * max_tokens

        Add num_tokens(prompt) if Request.echo_prompt is True.
        """
        tokenizer: Tokenizer = TokenizerFactory.get_tokenizer("openai", metric_service)
        num_prompt_tokens: int = tokenizer.tokenize_and_count(request.prompt)
        total_estimated_tokens: int = num_prompt_tokens + request.num_completions * request.max_tokens

        # We should add the number of tokens in the prompt twice when echo_prompt is True because OpenAI counts
        # both the tokens in the prompt and the completions, which in this case, the original prompt is included.
        if request.echo_prompt:
            total_estimated_tokens += num_prompt_tokens
        return total_estimated_tokens
