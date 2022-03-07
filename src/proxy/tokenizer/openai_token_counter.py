from typing import List

from common.request import Request, Sequence
from .token_counter import TokenCounter
from .openai_tokenizer import OpenAITokenizer


class OpenAITokenCounter(TokenCounter):

    # From https://help.openai.com/en/articles/5072518-controlling-the-length-of-completions,
    # "these requests can use up to 2,049 tokens, shared between prompt and completion."
    MAX_CONTEXT_TOKEN_LENGTH = 2049

    def __init__(self):
        self.tokenizer = OpenAITokenizer()

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the total number of tokens using the suggestion here:
        https://community.openai.com/t/how-do-i-calculate-the-pricing-for-generation-of-text/11662/5

        TODO: OpenAI will support counting the number of tokens for us. Adapt this method accordingly.
              https://github.com/stanford-crfm/benchmarking/issues/59
        """
        n_tokens: int = self.tokenize_and_count(request.model, request.prompt)
        for sequence in completions:
            n_tokens += len(sequence.tokens)
        return n_tokens

    def estimate_tokens(self, request: Request) -> int:
        """
        Estimate the number of tokens for a given request. Include the tokens in the prompt
        when estimating number of tokens. Formula:

            num_tokens(prompt) + num_completions * max_tokens

        Add num_tokens(prompt) if Request.echo_prompt is True.
        """
        num_tokens_in_prompt: int = self.tokenize_and_count(request.model, request.prompt)
        total_estimated_tokens: int = num_tokens_in_prompt + request.num_completions * request.max_tokens

        # We should add the number of tokens in the prompt twice when echo_prompt is True because OpenAI counts
        # both the tokens in the prompt and the completions, which in this case, the original prompt is included.
        if request.echo_prompt:
            total_estimated_tokens += num_tokens_in_prompt
        return total_estimated_tokens

    def tokenize_and_count(self, model: str, text: str) -> int:
        """Count the number of tokens for a given text using the GPT-2 tokenizer."""
        return len(self.tokenizer.encode(text))

    def fits_within_context_window(self, model: str, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the GPT-3 context window taking to account
        the expected completion length (defaults to 0).
        """
        return (
            self.tokenize_and_count(model, text) + expected_completion_token_length
            <= OpenAITokenCounter.MAX_CONTEXT_TOKEN_LENGTH
        )

    def truncate_from_right(self, model: str, text: str) -> str:
        """
        Truncates text from the right to fit within the GPT-3 context window.

        By default, HuggingFace uses the 'longest_first' truncation strategy:
        "Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one
        at each token (when there is a pair of input sequences)."

        Since we are only passing in a single string, the tokenizer will simply truncate from the right.
        """
        return self.tokenizer.decode(
            self.tokenizer.encode(text, truncation=True, max_length=OpenAITokenCounter.MAX_CONTEXT_TOKEN_LENGTH)
        )
