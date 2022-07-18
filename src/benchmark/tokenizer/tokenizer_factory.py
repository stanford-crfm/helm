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
        # TODO: create tokenizers for the individual offline models
        #       https://github.com/stanford-crfm/benchmarking/issues/588
        elif organization == "openai" or organization == "simple" or organization == "together":
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
