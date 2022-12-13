from typing import List

from helm.common.request import Request, Sequence
from .token_counter import TokenCounter


class CohereTokenCounter(TokenCounter):
    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the number of generated tokens.
        TODO: Cohere simply counts the number of generations, but we currently only support counting tokens.
        """
        return sum(len(sequence.tokens) for sequence in completions)
