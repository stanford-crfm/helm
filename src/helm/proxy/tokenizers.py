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
    
@dataclass(frozen=True)
class TokenizerSpec:

    # Name of the tokenizer (e.g. "huggingface/gpt2")
    # The format is "{organization}/{name}"
    name: str

    # The end of text token (e.g. "<|endoftext|>")
    end_of_text_token: str

    # The prefix token (e.g. "<|endoftext|>")
    prefix_token: str


    @property
    def tokenizer_organization(self):
        """Example: 'huggingface/gpt2' => 'huggingface'"""
        return self.name.split("/")[0]

    @property
    def tokenizer_name(self):
        """Example: 'huggingface/gpt2' => 'gpt2'"""
        return self.name.split("/")[1]
    

class Tokenizer(ABC):

    def __init__(self, spec: TokenizerSpec):
        self.spec: TokenizerSpec = spec

    @property
    def name(self) -> str:
        """The name of the tokenizer."""
        return self.spec.name

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return self.spec.end_of_text_token

    @property
    def prefix_token(self) -> str:
        """The prefix token"""
        return self.spec.prefix_token

    def get_num_tokens(self, text: str) -> int:
        """Tokenizes the text and counts the number of tokens."""
        result: EncodeResult = self.encode(text)
        return len(result.tokens)

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


ALL_TOKENIZERS = [
    TokenizerSpec(name="huggingface/gpt2", end_of_text_token="<|endoftext|>", prefix_token="<|endoftext|>"),
]