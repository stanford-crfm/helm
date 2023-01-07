from typing import Dict, List

from helm.common.request import Request, Sequence
from helm.proxy.clients.huggingface_client import HuggingFaceClient
from helm.proxy.models import is_text_to_image_model
from .ai21_token_counter import AI21TokenCounter
from .cohere_token_counter import CohereTokenCounter
from .free_token_counter import FreeTokenCounter
from .gooseai_token_counter import GooseAITokenCounter
from .image_counter import ImageCounter
from .openai_token_counter import OpenAITokenCounter
from .token_counter import TokenCounter


class AutoTokenCounter(TokenCounter):
    """Automatically count tokens based on the organization."""

    def __init__(self, huggingface_client: HuggingFaceClient):
        self.token_counters: Dict[str, TokenCounter] = {}
        self.huggingface_client: HuggingFaceClient = huggingface_client

    def _get_token_counter(self, request: Request) -> TokenCounter:
        """Return a token counter based on the request."""
        organization: str = request.model_organization
        token_counter = self.token_counters.get(organization)
        if token_counter is None:
            if is_text_to_image_model(request.model):
                token_counter = ImageCounter()
            elif organization == "openai":
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
        token_counter: TokenCounter = self._get_token_counter(request)
        return token_counter.count_tokens(request, completions)
