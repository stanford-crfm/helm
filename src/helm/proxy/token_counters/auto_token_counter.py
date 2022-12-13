from typing import Dict, List

from helm.common.request import Request, Sequence
from helm.proxy.clients.huggingface_client import HuggingFaceClient
from .ai21_token_counter import AI21TokenCounter
from .cohere_token_counter import CohereTokenCounter
from .free_token_counter import FreeTokenCounter
from .gooseai_token_counter import GooseAITokenCounter
from .openai_token_counter import OpenAITokenCounter
from .token_counter import TokenCounter


class AutoTokenCounter(TokenCounter):
    """Automatically count tokens based on the organization."""

    def __init__(self, huggingface_client: HuggingFaceClient):
        self.token_counters: Dict[str, TokenCounter] = {}
        self.huggingface_client: HuggingFaceClient = huggingface_client

    def get_token_counter(self, organization: str) -> TokenCounter:
        """Return a token counter based on the organization."""
        token_counter = self.token_counters.get(organization)
        if token_counter is None:
            if organization == "openai":
                token_counter = OpenAITokenCounter(self.huggingface_client)
            elif organization == "ai21":
                token_counter = AI21TokenCounter()
            elif organization == "gooseai":
                token_counter = GooseAITokenCounter()
            elif organization == "cohere":
                token_counter = CohereTokenCounter()
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
