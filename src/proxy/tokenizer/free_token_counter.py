from typing import List

from common.request import Request, Sequence
from proxy.tokenizer.token_counter import TokenCounter


class FreeTokenCounter(TokenCounter):
    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """For when we don't care about keeping track of the number of tokens."""
        return 0
