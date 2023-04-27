from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass

from helm.common.tokenization_request import TokenizationToken
from helm.proxy.tokenizers.tiktoken_tokenizer import TiktokenTokenizerModel
from helm.proxy.tokenizers.huggingface_tokenizer import HuggingFaceTokenizerModel


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

    # Name of the provider of the TokenizerModel to use (e.g. "huggingface")
    provider: str

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
    

class TokenizerModel(ABC):

    def __init__(self):
        self.spec: Optional[TokenizerSpec] = None

    def set_spec(self, spec: TokenizerSpec):
        self.spec = spec

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        The name of the provider (e.g. "huggingface", "tiktoken").
        The provider is not necessarily the same as the tokenizer organization.
        For example the provider for the openai tokenizer is "tiktoken".
        """
        pass

    @property
    def name(self) -> str:
        """The name of the tokenizer."""
        if self.spec is None:
            raise ValueError("TokenizerModel.spec is not set")
        return self.spec.name

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        if self.spec is None:
            raise ValueError("TokenizerModel.spec is not set")
        return self.spec.end_of_text_token

    @property
    def prefix_token(self) -> str:
        """The prefix token"""
        if self.spec is None:
            raise ValueError("TokenizerModel.spec is not set")
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

ALL_TOKENIZER_MODELS: List[TokenizerModel] = [
    # TODO: Figure out how to implement this list without initializing the models
    # maybe add a function model.set_spec(spec) that sets the spec and initializes the model
    HuggingFaceTokenizerModel(),
    TiktokenTokenizerModel(),
]

PROVIDER_NAME_TO_TOKENIZER_MODEL: Dict[str, TokenizerModel] = {model.provider_name: model for model in ALL_TOKENIZER_MODELS}


def get_tokenizer_model(provider_name: str) -> TokenizerModel:
    """Get the `TokenizerModel` given the provider name."""
    if provider_name not in PROVIDER_NAME_TO_TOKENIZER_MODEL:
        raise ValueError(f"No tokenizer model with provider name: {provider_name}")

    return PROVIDER_NAME_TO_TOKENIZER_MODEL[provider_name]


def get_all_tokenizer_models() -> List[str]:
    """Get all provider names of tokenizer models."""
    return list(PROVIDER_NAME_TO_TOKENIZER_MODEL.keys())


ALL_TOKENIZERS: List[TokenizerSpec] = [
    TokenizerSpec(provider="huggingface", name="huggingface/gpt2", end_of_text_token="<|endoftext|>", prefix_token="<|endoftext|>"),
    TokenizerSpec(provider="tiktoken", name="openai/cl100k_base", end_of_text_token="<|endoftext|>", prefix_token="<|endoftext|>"),
]

TOKENIZER_NAME_TO_SPEC: Dict[str, TokenizerSpec] = {tokenizer.name: tokenizer for tokenizer in ALL_TOKENIZERS}

def get_tokenizer_spec(tokenizer_name: str) -> TokenizerSpec:
    """Get the `TokenizerSpec` given the tokenizer name."""
    if tokenizer_name not in TOKENIZER_NAME_TO_SPEC:
        raise ValueError(f"No tokenizer spec with name: {tokenizer_name}")

    return TOKENIZER_NAME_TO_SPEC[tokenizer_name]

def get_all_tokenizer_specs() -> List[str]:
    """Get all tokenizer names."""
    return list(TOKENIZER_NAME_TO_SPEC.keys())

def get_tokenizer_from_spec(spec: TokenizerSpec) -> TokenizerModel:
    """Get the tokenizer model given the tokenizer spec."""
    return get_tokenizer_model(spec.provider)

def get_tokenizer(tokenizer_name: str) -> TokenizerModel:
    """Get the tokenizer model given the tokenizer name."""
    spec: TokenizerSpec = get_tokenizer_spec(tokenizer_name)
    model: TokenizerModel = get_tokenizer_from_spec(spec)
    model.set_spec(spec)
    return model