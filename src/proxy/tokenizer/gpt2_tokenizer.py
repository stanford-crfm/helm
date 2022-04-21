from typing import List, Optional
from transformers import GPT2TokenizerFast

from .tokenizer import Tokenizer, EncodeResult


class GPT2Tokenizer(Tokenizer):

    # The max request length of GPT2 is MAX_SEQUENCE_LENGTH + 1.
    MAX_REQUEST_LENGTH: int = 1025

    # The max length of the model input. The max sequence length for GPT-2 is 1024.
    MAX_SEQUENCE_LENGTH: int = 1024

    # The end of text token
    END_OF_TEXT_TOKEN: str = "<|endoftext|>"

    def __init__(self, tokenizer: GPT2TokenizerFast):
        self._tokenizer = tokenizer

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of this tokenizer."""
        return GPT2Tokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length of the OpenAI models."""
        return GPT2Tokenizer.MAX_REQUEST_LENGTH

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return GPT2Tokenizer.END_OF_TEXT_TOKEN

    @property
    def prefix_token(self) -> str:
        """The prefix token for OPENAI models is the end of text token."""
        return self.end_of_text_token

    def encode(self, text: str) -> EncodeResult:
        """
        Encodes the input text to tokens.
        """
        tokens: List = self._tokenizer.encode(text)
        return EncodeResult(text=text, tokens=tokens)

    def decode(self, tokens: List, text: Optional[str] = None) -> str:
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
        Checks if the given text fits within the context window given by `max_sequence_length
        taking to account the expected completion length (defaults to 0).
        """
        return self.tokenize_and_count(text) + expected_completion_token_length <= self.max_sequence_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_sequence_length`
        minus the expected completion length (defaults to 0).

        By default, HuggingFace uses the 'longest_first' truncation strategy:
        "Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one
        at each token (when there is a pair of input sequences)."

        Since we are only passing in a single string, the tokenizer will simply truncate from the right.
        """
        return self._tokenizer.decode(
            self._tokenizer.encode(
                text=text, truncation=True, max_length=self.max_sequence_length - expected_completion_token_length
            )
        )
