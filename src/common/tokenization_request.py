from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass(frozen=True)
class EncodeParameters:
    # Whether to truncate
    truncation: bool

    # Maximum length when encoding
    max_length: int


@dataclass(frozen=True)
class TokenizationRequest:
    """A `TokenizationRequest` specifies how to tokenize some text."""

    # Text to tokenize
    text: str

    # Which tokenizer we should use
    tokenizer: str = "huggingface/gpt2_tokenizer_fast"

    # For HuggingFace tokenizers
    encode_parameters: Optional[EncodeParameters] = None

    @property
    def tokenizer_organization(self):
        """Example: 'huggingface/gpt2_tokenizer_fast' => 'huggingface'"""
        return self.tokenizer.split("/")[0]


@dataclass(frozen=True)
class TextRange:
    """The range within the original text."""

    # Start position of the original text inclusive
    start: int

    # End position of the original text exclusive
    end: int


@dataclass(frozen=True)
class TokenizationToken:
    """Representation of a single token when tokenizing."""

    # Value of the token. Can be a string or integer.
    value: Union[str, int]

    # The text range the token was generated from.
    text_range: Optional[TextRange] = None


@dataclass(frozen=True)
class TokenizationRequestResult:
    """Result after sending a `TokenizationRequest`."""

    # Whether the request was successful
    success: bool

    # Whether the request was cached
    cached: bool

    # The input text transformed by the tokenizer for tokenization.
    # e.g. The AI21 tokenizer replaces "½" with "1/2" before tokenization.
    text: str

    # The list of tokens
    tokens: List[TokenizationToken]

    # How long did the tokenization take?
    request_time: Optional[float] = None

    # If `success` is false, what was the error?
    error: Optional[str] = None


@dataclass(frozen=True)
class DecodeRequest:
    """For HuggingFace tokenizers. How to decode tokens and convert it to text."""

    # Tokens
    tokens: List[int]

    # Which tokenizer we should use
    tokenizer: str = "huggingface/gpt2_tokenizer_fast"

    # Whether to clean up the tokenization spaces. Should be False to preserve the original text.
    clean_up_tokenization_spaces: bool = False

    @property
    def tokenizer_organization(self):
        """Example: 'huggingface/gpt2_tokenizer_fast' => 'huggingface'"""
        return self.tokenizer.split("/")[0]


@dataclass(frozen=True)
class DecodeRequestResult:
    """Result after sending a `DecodeRequest`."""

    # Whether the request was successful
    success: bool

    # Whether the request was cached
    cached: bool

    # The resulting text after decoding
    text: str

    # How long did the decoding take?
    request_time: Optional[float] = None

    # If `success` is false, what was the error?
    error: Optional[str] = None
