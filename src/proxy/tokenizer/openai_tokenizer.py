from typing import List
from transformers import GPT2TokenizerFast

from .tokenizer import Tokenizer


class OpenAITokenizer(Tokenizer):

    # From https://help.openai.com/en/articles/5072518-controlling-the-length-of-completions,
    # "these requests can use up to 2049 tokens, shared between prompt and completion."
    MAX_REQUEST_LENGTH: int = 2049

    # The max length of the model input. The max sequence length for OpenAI is 2048,
    # which is different from the max request length of 2049.
    MAX_SEQUENCE_LENGTH: int = 2048

    # The end of text token
    END_OF_TEXT_TOKEN: str = "<|endoftext|>"

    def __init__(self):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        # Weights are cached at ~/.cache/huggingface/transformers.
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the OpenAI models."""
        return OpenAITokenizer.MAX_SEQUENCE_LENGTH

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return OpenAITokenizer.END_OF_TEXT_TOKEN

    def encode(self, text: str) -> List[int]:
        """
        Encodes the input text to tokens.
        """
        return self._tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Given a list of tokens, outputs the corresponding text.
        """
        return self._tokenizer.decode(tokens, clean_up_tokenization_spaces=False)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text.
        """
        return self._tokenizer.tokenize(text)

    def tokenize_and_count(self, text: str) -> int:
        """Tokenizes the text using the GPT-2 tokenizer and returns the number of tokens."""
        return len(self.tokenize(text))

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the GPT-3 context window taking to account
        the expected completion length (defaults to 0).
        """
        return self.tokenize_and_count(text) + expected_completion_token_length <= OpenAITokenizer.MAX_REQUEST_LENGTH

    def truncate_from_right(self, text: str) -> str:
        """
        Truncates text from the right to fit within the GPT-3 context window.

        By default, HuggingFace uses the 'longest_first' truncation strategy:
        "Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one
        at each token (when there is a pair of input sequences)."

        Since we are only passing in a single string, the tokenizer will simply truncate from the right.
        """
        return self._tokenizer.decode(
            self._tokenizer.encode(text, truncation=True, max_length=OpenAITokenizer.MAX_REQUEST_LENGTH)
        )
