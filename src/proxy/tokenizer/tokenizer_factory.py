from typing import Dict, Optional, Type

from common.hierarchical_logger import htrack_block
from benchmark.tokenizer_service import TokenizerService
from .ai21_tokenizer import AI21Tokenizer
from .tokenizer import Tokenizer
from .openai_tokenizer import OpenAITokenizer
from .gpt2_tokenizer import GPT2Tokenizer
from .gptj_tokenizer import GPTJTokenizer


class TokenizerFactory:
    # Cached tokenizers
    cached_tokenizers: Dict[str, Tokenizer] = {}

    @staticmethod
    def get_tokenizer(model: str, service: Optional[TokenizerService] = None) -> Tokenizer:
        """
        Returns a `Tokenizer` given the `model`, creating one if necessary.
        Make sure this function returns instantaneously on repeated calls.
        """
        organization: str = model.split("/")[0]

        tokenizer: Tokenizer
        if organization == "openai" or organization == "simple":
            tokenizer = TokenizerFactory.create_or_get_cached_tokenizer(model, OpenAITokenizer)
        elif model == "huggingface/gpt2":
            tokenizer = TokenizerFactory.create_or_get_cached_tokenizer(model, GPT2Tokenizer)
        elif model == "huggingface/gptj_6b":
            tokenizer = TokenizerFactory.create_or_get_cached_tokenizer(model, GPTJTokenizer)
        elif organization == "ai21":
            if not service:
                raise ValueError("Need to pass in a TokenizerService to get the tokenizer for the AI21 models.")

            # Don't need to cache since AI21Tokenizer is just a wrapper.
            tokenizer = AI21Tokenizer(model=model, service=service)
        else:
            raise ValueError(f"Unsupported model: {model}")

        return tokenizer

    @staticmethod
    def create_or_get_cached_tokenizer(model: str, tokenizer_class: Type[Tokenizer]) -> Tokenizer:
        """
        Checks if the tokenizer is cached for the specific model. Creates the tokenizer
        if it's not cached. Returns the tokenizer.
        """
        if model not in TokenizerFactory.cached_tokenizers:
            with htrack_block(f"Creating {tokenizer_class.__name__} tokenizer for {model}"):
                TokenizerFactory.cached_tokenizers[model] = tokenizer_class()
        return TokenizerFactory.cached_tokenizers[model]
