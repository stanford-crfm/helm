from typing import List

from helm.common.request import Request, Sequence
from .token_counter import TokenCounter


class GooseAITokenCounter(TokenCounter):
    # From https://goose.ai/pricing: "the base price includes your first 25 tokens
    # generated, and you can scale beyond that on a per-token basis."
    BASE_PRICE_TOKENS: int = 25

    @staticmethod
    def account_for_base_tokens(num_tokens: int):
        """Subtracts the number of tokens included in the base price."""
        return max(num_tokens - GooseAITokenCounter.BASE_PRICE_TOKENS, 0)

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the number of generated tokens and NOT the number of tokens in the prompt.
        From https://goose.ai/pricing: "by charging only for output, you have control since
        you can configure the maximum number of tokens generated per API call
        (up to 2,048 tokens)."
        """
        return GooseAITokenCounter.account_for_base_tokens(sum(len(sequence.tokens) for sequence in completions))
