from typing import List

from common.request import Request, Sequence
from proxy.tokenizer.token_counter import TokenCounter


class FreeTokenCounter(TokenCounter):
    """For when we don't care about keeping track of the number of tokens."""

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """No need to count tokens, since it's free. Return 0."""
        return 0

    def estimate_tokens(self, request: Request) -> int:
        """No need to estimate tokens, since it's free. Return 0."""
        return 0

    def tokenize_and_count(self, model: str, text: str) -> int:
        """No need to count tokens for the given text, since it's free. Return 0."""
        return 0

    def fits_within_context_window(self, model: str, text: str, expected_completion_token_length: int) -> bool:
        """There is no constraint on the context window size. Everything fits."""
        return True

    def truncate_from_right(self, model: str, text: str) -> str:
        """There is no constraint on the context window size. Return the original text."""
        return text
