from abc import ABC, abstractmethod
from typing import List, Optional

from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes `request.text` using `request.tokenizer`.
        Returns a `TokenizationRequestResult` object.
        """
        pass

    @abstractmethod
    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes `request.tokens` using `request.tokenizer`.
        Returns a `DecodeRequestResult` object.
        """
        pass


def cleanup_str(token: str, tokenizer_name: Optional[str] = None) -> str:
    """
    Certain tokenizers introduce special characters to represent spaces, such as
    "Ġ" or "▁". This function removes those characters.
    """
    if tokenizer_name in [
        "TsinghuaKEG/ice",
        "bigscience/T0pp",
        "google/t5-11b",
        "google/flan-t5-xxl",
        "google/ul2",
        "Yandex/yalm",
        "ai21/j1",
        "together",
    ]:
        return token.replace("▁", " ")
    elif tokenizer_name is not None and (
        tokenizer_name.startswith("huggingface") or tokenizer_name == "anthropic/claude"
    ):
        return token.replace("Ġ", " ")
    return token


def cleanup_tokens(tokens: List[str], tokenizer_name: Optional[str] = None) -> List[str]:
    """
    Applies `cleanup_str` to each token in `tokens`.
    """
    return [cleanup_str(token, tokenizer_name) for token in tokens]
