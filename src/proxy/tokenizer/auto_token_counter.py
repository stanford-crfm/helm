from typing import Dict, List

from common.request import Request, Sequence
from proxy.tokenizer.ai21_token_counter import AI21TokenCounter
from proxy.tokenizer.free_token_counter import FreeTokenCounter
from proxy.tokenizer.openai_token_counter import OpenAITokenCounter
from proxy.tokenizer.token_counter import TokenCounter


class AutoTokenCounter(TokenCounter):
    """Automatically count tokens based on the organization."""

    def __init__(self):
        self.token_counters: Dict[str, TokenCounter] = {}

    def get_token_counter(self, organization: str) -> TokenCounter:
        """Return a token counter based on the organization."""
        token_counter = self.token_counters.get(organization)
        if token_counter is None:
            # TODO: implement token counter for HuggingFace
            #       https://github.com/stanford-crfm/benchmarking/issues/6
            if organization == "openai":
                token_counter = OpenAITokenCounter()
            elif organization == "ai21":
                token_counter = AI21TokenCounter()
            else:
                token_counter = FreeTokenCounter()
            self.token_counters[organization] = token_counter
        return token_counter

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts tokens based on the organization.
        """
        token_counter: TokenCounter = self.get_token_counter(request.model_organization)
        return token_counter.count_tokens(request, completions)
