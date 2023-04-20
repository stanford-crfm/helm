from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from helm.common.tokenization_request import TokenizationToken


@dataclass(frozen=True)
class EncodeResult:
    """Result returned by the encode() method."""

    # The input text transformed by the tokenizer for tokenization.
    # e.g. The AI21 tokenizer replaces "½" with "1/2" before tokenization.
    text: str

    # The resulting list of `TokenizationToken`s after encoding
    tokens: List[TokenizationToken]

    @property
    def token_values(self):
        return [token.value for token in self.tokens]


class WindowService(ABC):
    @property
    @abstractmethod
    def tokenizer_name(self) -> str:
        pass

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """The max length of the model input."""
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
    @abstractmethod
    def prefix_token(self) -> str:
        """The prefix token"""
        pass

    @abstractmethod
    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        """Encodes the input text to tokens given the model"""
        pass

    @abstractmethod
    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        """
        Given the model and a list of tokens, outputs the corresponding text.

        For models using the GPT-2 tokenizer, the tokens are integers; for AI21
        models, the tokens are `TokenizationToken`s.

        Some tokenizers (e.g. AI21) normalize the text before encoding it and
        thus require the `normalized_text` for decoding.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes the text."""
        pass

    @abstractmethod
    def get_num_tokens(self, text: str) -> int:
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
