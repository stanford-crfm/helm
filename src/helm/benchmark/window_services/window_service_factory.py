from helm.proxy.models import get_model, get_model_names_with_tag, Model, WIDER_CONTEXT_WINDOW_TAG
from .ai21_window_service import AI21WindowService
from .anthropic_window_service import AnthropicWindowService
from .cohere_window_service import CohereWindowService
from .luminous_window_service import (
    LuminousBaseWindowService,
    LuminousExtendedWindowService,
    LuminousSupremeWindowService,
    LuminousWorldWindowService,
)
from .openai_window_service import OpenAIWindowService
from .wider_openai_window_service import WiderOpenAIWindowService
from .mt_nlg_window_service import MTNLGWindowService
from .bloom_window_service import BloomWindowService
from .ice_window_service import ICEWindowService
from .gpt2_window_service import GPT2WindowService
from .gptj_window_service import GPTJWindowService
from .gptneox_window_service import GPTNeoXWindowService
from .opt_window_service import OPTWindowService
from .t0pp_window_service import T0ppWindowService
from .t511b_window_service import T511bWindowService
from .ul2_window_service import UL2WindowService
from .yalm_window_service import YaLMWindowService
from .window_service import WindowService
from .tokenizer_service import TokenizerService


class WindowServiceFactory:
    @staticmethod
    def get_window_service(model_name: str, service: TokenizerService) -> WindowService:
        """
        Returns a `WindowService` given the name of the model.
        Make sure this function returns instantaneously on repeated calls.
        """
        model: Model = get_model(model_name)
        organization: str = model.organization
        engine: str = model.engine

        window_service: WindowService
        if model_name in get_model_names_with_tag(WIDER_CONTEXT_WINDOW_TAG):
            window_service = WiderOpenAIWindowService(service)
        elif organization == "openai" or organization == "simple":
            window_service = OpenAIWindowService(service)
        elif organization == "AlephAlpha":
            if engine == "luminous-base":
                window_service = LuminousBaseWindowService(service)
            elif engine == "luminous-extended":
                window_service = LuminousExtendedWindowService(service)
            elif engine == "luminous-supreme":
                window_service = LuminousSupremeWindowService(service)
            elif engine == "luminous-world":
                window_service = LuminousWorldWindowService(service)
            else:
                raise ValueError(f"Unhandled Aleph Alpha model: {engine}")
        elif organization == "microsoft":
            window_service = MTNLGWindowService(service)
        elif organization == "anthropic":
            window_service = AnthropicWindowService(service)
        elif model_name == "huggingface/gpt2":
            window_service = GPT2WindowService(service)
        elif model_name == "together/bloom":
            window_service = BloomWindowService(service)
        elif model_name == "together/glm":
            # From https://github.com/THUDM/GLM-130B, "the tokenizer is implemented based on
            # icetk---a unified multimodal tokenizer for images, Chinese, and English."
            window_service = ICEWindowService(service)
        elif model_name in ["huggingface/gpt-j-6b", "together/gpt-j-6b", "gooseai/gpt-j-6b"]:
            window_service = GPTJWindowService(service)
        elif model_name in ["together/gpt-neox-20b", "gooseai/gpt-neo-20b"]:
            window_service = GPTNeoXWindowService(service)
        elif model_name in ["together/opt-66b", "together/opt-175b"]:
            window_service = OPTWindowService(service)
        elif model_name == "together/t0pp":
            window_service = T0ppWindowService(service)
        elif model_name == "together/t5-11b":
            window_service = T511bWindowService(service)
        elif model_name == "together/ul2":
            window_service = UL2WindowService(service)
        elif model_name == "together/yalm":
            window_service = YaLMWindowService(service)
        elif organization == "cohere":
            window_service = CohereWindowService(service)
        elif organization == "ai21":
            window_service = AI21WindowService(service=service, gpt2_window_service=GPT2WindowService(service))
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        return window_service
