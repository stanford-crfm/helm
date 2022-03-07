from typing import Dict, List

from .openai_tokenizer import OpenAITokenizer
from .tokenizer import Tokenizer


class AutoTokenizer:
    """Automatically tokenizes inputs based on the organization."""

    def __init__(self):
        self.tokenizers: Dict[str, Tokenizer] = {}

    def get_tokenizer(self, model: str) -> Tokenizer:
        """Return a tokenizer based on the organization."""
        organization = model.split("/")[0]
        tokenizer = self.tokenizers.get(organization)
        if tokenizer is None:
            if organization == "openai":
                tokenizer = OpenAITokenizer()
            else:
                raise Exception(f"Unsupported model: {model}")
            self.tokenizers[organization] = tokenizer
        return tokenizer

    def encode(self, text: str, model: str) -> List[int]:
        """
        Encodes the input text to tokens.
        """
        model_tokenizer = self.get_tokenizer(model)
        return model_tokenizer.encode(text)

    def decode(self, tokens: List[int], model: str) -> str:
        """
        Given a list of tokens, outputs the corresponding text.
        """
        model_tokenizer = self.get_tokenizer(model)
        return model_tokenizer.decode(tokens)

    def tokenize(self, text: str, model: str) -> List[str]:
        """
        Given a list of tokens, outputs the corresponding text.
        """
        model_tokenizer = self.get_tokenizer(model)
        return model_tokenizer.tokenize(text)
