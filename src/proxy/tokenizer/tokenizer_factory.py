from typing import Optional

from common.hierarchical_logger import htrack_block
from benchmark.tokenizer_service import TokenizerService
from .ai21_tokenizer import AI21Tokenizer
from .tokenizer import Tokenizer
from .openai_tokenizer import OpenAITokenizer


class TokenizerFactory:
    # Cached tokenizers
    openai_tokenizer: Optional[Tokenizer] = None

    @staticmethod
    def get_tokenizer(model: str, service: TokenizerService) -> Tokenizer:
        """
        Returns a `Tokenizer` given the `model`, creating one if necessary.
        Make sure this function returns instantaneously on repeated calls.
        """
        organization: str = model.split("/")[0]

        tokenizer: Tokenizer
        if organization == "openai" or organization == "simple":
            if TokenizerFactory.openai_tokenizer is None:
                with htrack_block("Creating OpenAITokenizer"):
                    TokenizerFactory.openai_tokenizer = OpenAITokenizer()
            tokenizer = TokenizerFactory.openai_tokenizer
        elif organization == "ai21":
            # Don't need to cache since AI21Tokenizer is just a wrapper.
            tokenizer = AI21Tokenizer(model=model, service=service)
        else:
            raise ValueError(f"Unsupported model: {model}")

        return tokenizer
