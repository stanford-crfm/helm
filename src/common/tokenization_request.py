from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class TokenizationRequest:
    """A `TokenizationRequest` specifies how to tokenize some text."""

    # Text to tokenize
    text: str

    # Which model whose tokenizer we should use
    model: str = "openai/davinci"

    @property
    def model_organization(self):
        """Example: 'ai21/j1-jumbo' => 'ai21'"""
        return self.model.split("/")[0]


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

    # Text of the token
    text: str

    # The text range the token was generated from.
    text_range: Optional[TextRange] = None


@dataclass(frozen=True)
class TokenizationRequestResult:
    """Result after sending a `TokenizationRequest`."""

    # Whether the request was cached
    cached: bool

    # The input text transformed by the tokenizer for tokenization.
    # e.g. The AI21 tokenizer replaces "½" with "1/2" before tokenization.
    text: str

    # The list of tokens
    tokens: List[TokenizationToken]
