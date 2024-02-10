from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from helm.common.tokenization_request import TokenizationToken

INT_MAX: int = 2**31 - 1


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
    def max_sequence_and_generated_tokens_length(self) -> int:
        """
        The max length of the model input and output tokens.
        Some models (like Anthropic/Claude and Megatron) have a
        specifix limit sequence length + max_token.

        Since models do not have this limit, returns INT_MAX.
        """
        return INT_MAX

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


class ConfigurableWindowService(WindowService, ABC):
    def __init__(
        self,
        tokenizer_name: str,
        max_sequence_length: int,
        max_request_length: Optional[int] = None,
        max_sequence_and_generated_tokens_length: Optional[int] = None,
        end_of_text_token: Optional[str] = None,
        prefix_token: Optional[str] = None,
    ):
        self._tokenizer_name = tokenizer_name
        self._max_sequence_length = max_sequence_length
        self._max_request_length = max_request_length or max_sequence_length
        self._max_sequence_and_generated_tokens_length = max_sequence_and_generated_tokens_length or INT_MAX
        self._end_of_text_token = end_of_text_token or ""
        self._prefix_token = prefix_token or ""

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def max_request_length(self) -> int:
        return self._max_request_length

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self._max_sequence_and_generated_tokens_length

    @property
    def end_of_text_token(self) -> str:
        return self._end_of_text_token

    @property
    def prefix_token(self) -> str:
        return self._prefix_token
