from typing import List

from common.request import Request, Sequence
from common.tokenizer.token_counter import TokenCounter


class FreeTokenCounter(TokenCounter):
    """For when we don't care about keeping track of the number of tokens."""

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """No need to count tokens, since it's free. Return 0."""
        return 0

    def estimate_tokens(self, request: Request) -> int:
        """No need to estimate tokens, since it's free. Return 0."""
        return 0
