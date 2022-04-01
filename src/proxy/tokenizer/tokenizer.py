from abc import ABC, abstractmethod
from typing import List, Optional


class Tokenizer(ABC):
    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """ The max length of the model input."""
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
    def encode(self, text: str) -> List:
        """Encodes the input text to tokens given the model"""
        pass

    @abstractmethod
    def decode(self, tokens: List, original_text: Optional[str] = None) -> str:
        """Given the model and a list of tokens, outputs the corresponding text."""
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
