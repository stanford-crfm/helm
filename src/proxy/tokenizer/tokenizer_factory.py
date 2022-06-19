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
    def get_tokenizer(model_name_or_organization: str, service: TokenizerService) -> Tokenizer:
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
            tokenizer = WiderContextWindowOpenAITokenizer(service)
        elif organization == "openai" or organization == "simple":
            tokenizer = OpenAITokenizer(service)
        elif organization == "microsoft":
            tokenizer = MTNLGTokenizer(service)
        elif organization == "gooseai":
            tokenizer = GPTJTokenizer(service)
        elif organization == "anthropic":
            tokenizer = AnthropicTokenizer(service)
        elif model_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer(service)
        elif model_name == "huggingface/gptj_6b":
            tokenizer = GPTJTokenizer(service)
        elif organization == "ai21":
            tokenizer = AI21Tokenizer(service=service, gpt2_tokenizer=GPT2Tokenizer(service))
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
