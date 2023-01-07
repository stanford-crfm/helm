from typing import List

from helm.common.request import Request, Sequence
from .token_counter import TokenCounter


class ImageCounter(TokenCounter):
    """For text-to-image models."""

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """Treat each generated image as one token."""
        return len(completions)
