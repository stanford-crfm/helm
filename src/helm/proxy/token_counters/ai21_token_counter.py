from typing import List

from helm.common.request import Request, Sequence
from .token_counter import TokenCounter


class AI21TokenCounter(TokenCounter):
    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the number of generated tokens and NOT the number of tokens in the prompt
        (https://studio.ai21.com/docs/calculating-usage).

        The AI21 documentation (https://studio.ai21.com/docs/calculating-usage/) defines
        generated tokens as:
        "the total number of all completion tokens you generate. For example, assume you post
        a complete request for J1-Jumbo with a prompt consisting of 10 tokens and requiring 3
        completions, i.e. numResults = 3, and the model generates completions with 5, 15, and
        20 tokens. In total this request will consume 5+15+20=40 generated tokens."
        """
        return sum(len(sequence.tokens) for sequence in completions)
