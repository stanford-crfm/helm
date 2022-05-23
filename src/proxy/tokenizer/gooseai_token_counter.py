from typing import List

from common.request import Request, Sequence
from .token_counter import TokenCounter
from .tokenizer_factory import TokenizerFactory


class GooseAITokenCounter(TokenCounter):
    # From https://goose.ai/pricing: "the base price includes your first 25 tokens
    # generated, and you can scale beyond that on a per-token basis."
    _BASE_PRICE_TOKENS: int = 25

    @staticmethod
    def account_for_base_tokens(num_tokens: int):
        """Subtracts the number of tokens included in the base price."""
        return max(num_tokens - GooseAITokenCounter._BASE_PRICE_TOKENS, 0)

    def __init__(self):
        self.tokenizer = TokenizerFactory.get_tokenizer("gooseai")

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the number of generated tokens and NOT the number of tokens in the prompt.
        From https://goose.ai/pricing: "by charging only for output, you have control since
        you can configure the maximum number of tokens generated per API call
        (up to 2,048 tokens)."
        """
        return GooseAITokenCounter.account_for_base_tokens(sum(len(sequence.tokens) for sequence in completions))

    def estimate_tokens(self, request: Request) -> int:
        """
        Estimate the number of generated tokens for a given request. Formula:

            num_completions * max_tokens

        Add num_tokens(prompt) if `Request.echo_prompt` is True.
        """
        total_estimated_tokens: int = request.num_completions * request.max_tokens
        if request.echo_prompt:
            total_estimated_tokens += self.tokenizer.tokenize_and_count(request.prompt)
        return GooseAITokenCounter.account_for_base_tokens(total_estimated_tokens)
