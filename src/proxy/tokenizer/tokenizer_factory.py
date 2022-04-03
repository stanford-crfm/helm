from typing import Optional

from transformers import GPT2TokenizerFast

from common.hierarchical_logger import htrack_block
from .ai21_tokenizer import AI21Tokenizer
from .openai_tokenizer import OpenAITokenizer
from .gpt2_tokenizer import GPT2Tokenizer
from .gptj_tokenizer import GPTJTokenizer
from .tokenizer import Tokenizer
from .tokenizer_service import TokenizerService


class TokenizerFactory:
    # The underlying HuggingFace tokenizer
    gpt2_tokenizer_fast: Optional[GPT2TokenizerFast] = None

    @staticmethod
    def get_tokenizer(model: str, service: Optional[TokenizerService] = None) -> Tokenizer:
        """
        Returns a `Tokenizer` given the `model`,
        Make sure this function returns instantaneously on repeated calls.
        """
        organization: str = model.split("/")[0]

        tokenizer: Tokenizer
        if organization == "openai" or organization == "simple":
            tokenizer = OpenAITokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        # TODO: separate tokenizer for Anthropic? -Tony
        elif model == "huggingface/gpt2" or organization == "anthropic":
            tokenizer = GPT2Tokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif model == "huggingface/gptj_6b":
            tokenizer = GPTJTokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif organization == "ai21":
            if not service:
                raise ValueError("Need to pass in a TokenizerService to get the tokenizer for the AI21 models.")

            # Don't need to cache since AI21Tokenizer is just a wrapper.
            tokenizer = AI21Tokenizer(model=model, service=service)
        else:
            raise ValueError(f"Unsupported model: {model}")

        return tokenizer

    @staticmethod
    def get_gpt2_tokenizer_fast() -> GPT2TokenizerFast:
        """
        Checks if the underlying GPT-2 tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """
        if TokenizerFactory.gpt2_tokenizer_fast is None:
            # Weights are cached at ~/.cache/huggingface/transformers.
            with htrack_block("Creating GPT2TokenizerFast with Hugging Face Transformers"):
                TokenizerFactory.gpt2_tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
        return TokenizerFactory.gpt2_tokenizer_fast
