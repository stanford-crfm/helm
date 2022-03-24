from typing import Dict

from benchmark.tokenizer_service import TokenizerService
from .ai21_tokenizer import AI21Tokenizer
from .tokenizer import Tokenizer
from .openai_tokenizer import OpenAITokenizer


class TokenizerFactory:
    @staticmethod
    def create_tokenizer(model: str, service: TokenizerService) -> Tokenizer:
        """Creates a `Tokenizer` given the model."""
        organization: str = model.split("/")[0]

        tokenizer: Tokenizer
        if organization == "openai" or organization == "simple":
            tokenizer = OpenAITokenizer()
        elif organization == "ai21":
            tokenizer = AI21Tokenizer(model=model, service=service)
        else:
            raise ValueError(f"Unsupported model: {model}")

        return tokenizer

    def __init__(self):
        self.tokenizers: Dict[str, Tokenizer] = {}

    def get_tokenizer(self, model: str, service: TokenizerService) -> Tokenizer:
        """Caches and returns a `Tokenizer` given the model."""
        organization: str = model.split("/")[0]

        if organization not in self.tokenizers:
            # If the tokenizer for a specific organization is not found, create and cache it.
            tokenizer: Tokenizer = TokenizerFactory.create_tokenizer(model, service)
            self.tokenizers[organization] = tokenizer
        return self.tokenizers[organization]
