from typing import Optional

from transformers import GPT2TokenizerFast

from common.hierarchical_logger import htrack_block
from proxy.models import get_model, get_model_names_with_tag, Model, WIDER_CONTEXT_WINDOW_TAG
from .ai21_tokenizer import AI21Tokenizer
from .anthropic_tokenizer import AnthropicTokenizer
from .openai_tokenizer import OpenAITokenizer
from .wider_context_window_openai_tokenizer import WiderContextWindowOpenAITokenizer
from .mt_nlg_tokenizer import MTNLGTokenizer
from .gpt2_tokenizer import GPT2Tokenizer
from .gptj_tokenizer import GPTJTokenizer
from .tokenizer import Tokenizer
from .tokenizer_service import TokenizerService


class TokenizerFactory:
    # The underlying HuggingFace tokenizer
    gpt2_tokenizer_fast: Optional[GPT2TokenizerFast] = None

    @staticmethod
    def get_tokenizer(model_name_or_organization: str, service: Optional[TokenizerService] = None) -> Tokenizer:
        """
        Returns a `Tokenizer` given the model name or organization,
        Can pass in "openai" or "openai/davinci" for `model_name_or_organization`.
        Make sure this function returns instantaneously on repeated calls.
        """
        model_name: str = ""
        organization: str
        try:
            model: Model = get_model(model_name_or_organization)
            model_name = model.name
            organization = model.organization
        except ValueError:
            # At this point, an organization was passed in
            organization = model_name_or_organization

        tokenizer: Tokenizer
        if model_name in get_model_names_with_tag(WIDER_CONTEXT_WINDOW_TAG):
            tokenizer = WiderContextWindowOpenAITokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif organization == "openai" or organization == "simple":
            tokenizer = OpenAITokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif organization == "microsoft":
            tokenizer = MTNLGTokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif organization == "gooseai":
            tokenizer = GPTJTokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif organization == "anthropic":
            tokenizer = AnthropicTokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif model_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif model_name == "huggingface/gptj_6b":
            tokenizer = GPTJTokenizer(tokenizer=TokenizerFactory.get_gpt2_tokenizer_fast())
        elif organization == "ai21":
            if not service:
                raise ValueError("Need to pass in a TokenizerService to get the tokenizer for the AI21 models.")

            # Don't need to cache since AI21Tokenizer is just a wrapper.
            tokenizer = AI21Tokenizer(model=model_name, service=service)
        else:
            raise ValueError(f"Invalid model name or organization: {model_name_or_organization}")

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
