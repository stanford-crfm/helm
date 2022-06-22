from typing import List

from common.request import Request, Sequence
from common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from proxy.huggingface_client import HuggingFaceClient
from .token_counter import TokenCounter


class OpenAITokenCounter(TokenCounter):
    def __init__(self, huggingface_client: HuggingFaceClient):
        self.huggingface_client: HuggingFaceClient = huggingface_client

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the total number of tokens using the suggestion here:
        https://community.openai.com/t/how-do-i-calculate-the-pricing-for-generation-of-text/11662/5
        """
        tokenized_prompt: TokenizationRequestResult = self.huggingface_client.tokenize(
            TokenizationRequest(request.prompt)
        )
        # Number of tokens in the prompt + number of tokens in all the completions
        return len(tokenized_prompt.tokens) + sum([len(sequence.tokens) for sequence in completions])
