from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodeResult:
    """Result returned by the encode() method."""

    # The input text transformed by the tokenizer for tokenization.
    # e.g. The AI21 tokenizer replaces "½" with "1/2" before tokenization.
    text: str

    # The list of tokens. The tokens can be of any datatype as long as
    # they can be used for the decode() method. e.g. int for GPT-3 and
    # TokenizationToken for AI21.
    tokens: List


class Tokenizer(ABC):
    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """ The max length of the model input."""
        pass

    @property
    @abstractmethod
    def max_request_length(self) -> int:
        """
        The max request length of the model. Some models allow `max_request_length > max_sequence_length`
        so that users can specify the last output token. e.g. GPT-3.
        """
        pass

    @property
    @abstractmethod
    def end_of_text_token(self) -> str:
        """The end of text token."""
        pass

    @property
    def prefix_token(self) -> str:
        """The prefix token"""
        pass

    @abstractmethod
    def encode(self, text: str) -> EncodeResult:
        """Encodes the input text to tokens given the model"""
        pass

    @abstractmethod
    def decode(self, tokens: List, normalized_text: Optional[str] = None) -> str:
        """
        Given the model and a list of tokens, outputs the corresponding text.
        Some tokenizers (e.g. AI21) normalize the text before encoding it and
        require the `normalized_text` for decoding.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes the text."""
        pass

    @abstractmethod
    def tokenize_and_count(self, text: str) -> int:
        """Tokenizes the text and counts the number of tokens."""
        pass

    @abstractmethod
    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Whether the given text fits within the context window given the model and
        expected token length of the completion.
        """
        pass

    @abstractmethod
    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the given model's context window
        minus the expected completion length (defaults to 0).
        """
        pass
