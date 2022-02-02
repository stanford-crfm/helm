from typing import List

from common.request import Request, Sequence
from .token_counter import TokenCounter
from transformers import GPT2TokenizerFast


class OpenAITokenCounter(TokenCounter):
    # TODO: Add link to documentation
    MAX_CONTEXT_TOKEN_LENGTH = 2049

    def __init__(self):
        # OpenAI used the same tokenizer for GPT-2 and GPT-3.
        # Weights are cached at ~/.cache/huggingface/transformers.
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the total number of tokens using the suggestion here:
        https://community.openai.com/t/how-do-i-calculate-the-pricing-for-generation-of-text/11662/5

        TODO: OpenAI will support counting the number of tokens for us. Adapt this method accordingly.
              https://github.com/stanford-crfm/benchmarking/issues/59
        """
        n_tokens: int = self.tokenize_and_count(request.prompt)
        for sequence in completions:
            n_tokens += len(sequence.tokens)
        return n_tokens

    def estimate_tokens(self, request: Request) -> int:
        """
        Estimate the number of tokens for a given request. Include the tokens in the prompt
        when estimating number of tokens. Formula:

            num_tokens(prompt) + num_completions * max_tokens
        """
        return self.tokenize_and_count(request.prompt) + request.num_completions * request.max_tokens

    def tokenize_and_count(self, text: str) -> int:
        """Count the number of tokens for a given text using the GPT-2 tokenizer."""
        return len(self.tokenizer.encode(text))

    def truncate(self, text: str) -> str:
        """Tokenizes and truncates text to fit within the GPT-3 context window."""
        tokens: List[int] = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[: OpenAITokenCounter.MAX_CONTEXT_TOKEN_LENGTH])
