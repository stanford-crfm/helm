from typing import List

from helm.common.request import Request, Sequence
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.proxy.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer
from .token_counter import TokenCounter


class OpenAITokenCounter(TokenCounter):
    def __init__(self, huggingface_tokenizer: HuggingFaceTokenizer):
        self.huggingface_tokenizer: HuggingFaceTokenizer = huggingface_tokenizer

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the total number of tokens using the suggestion here:
        https://community.openai.com/t/how-do-i-calculate-the-pricing-for-generation-of-text/11662/5
        """
        tokenized_prompt: TokenizationRequestResult = self.huggingface_tokenizer.tokenize(
            TokenizationRequest(request.prompt)
        )
        # Number of tokens in the prompt + number of tokens in all the completions
        return len(tokenized_prompt.tokens) + sum([len(sequence.tokens) for sequence in completions])
