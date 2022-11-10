from typing import Dict, Callable

from proxy.models import ALL_MODELS, WIDER_CONTEXT_WINDOW_TAG
from .ai21_window_service import AI21WindowService
from .anthropic_window_service import AnthropicWindowService
from .cohere_window_service import CohereWindowService
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


WindowServiceFactoryCallable = Callable[[TokenizerService], WindowService]
# Dict of model name to window service factory
_window_service_factory_registry: Dict[str, WindowServiceFactoryCallable] = {}


def register_window_service_for_model(model_name: str, factory: WindowServiceFactoryCallable) -> None:
    if model_name in _window_service_factory_registry:
        raise ValueError(f"A window service factory is already registered for model name {model_name}")
    _window_service_factory_registry[model_name] = factory


def _register_build_in_window_services():
    for model in ALL_MODELS:
        model_name: str = model.name
        organization: str = model.organization
        if WIDER_CONTEXT_WINDOW_TAG in model.tags:
            register_window_service_for_model(model_name, WiderOpenAIWindowService)
        elif organization == "openai" or organization == "simple":
            register_window_service_for_model(model_name, OpenAIWindowService)
        elif organization == "microsoft":
            register_window_service_for_model(model_name, MTNLGWindowService)
        elif organization == "anthropic":
            register_window_service_for_model(model_name, AnthropicWindowService)
        elif model_name == "huggingface/gpt2":
            register_window_service_for_model(model_name, GPT2WindowService)
        elif model_name == "together/bloom":
            register_window_service_for_model(model_name, BloomWindowService)
        elif model_name == "together/glm":
            # From https://github.com/THUDM/GLM-130B, "the tokenizer is implemented based on
            # icetk---a unified multimodal tokenizer for images, Chinese, and English."
            register_window_service_for_model(model_name, ICEWindowService)
        elif model_name in ["huggingface/gpt-j-6b", "together/gpt-j-6b", "gooseai/gpt-j-6b"]:
            register_window_service_for_model(model_name, GPTJWindowService)
        elif model_name in ["together/gpt-neox-20b", "gooseai/gpt-neo-20b"]:
            register_window_service_for_model(model_name, GPTNeoXWindowService)
        elif model_name in ["together/opt-66b", "together/opt-175b"]:
            register_window_service_for_model(model_name, OPTWindowService)
        elif model_name == "together/t0pp":
            register_window_service_for_model(model_name, T0ppWindowService)
        elif model_name == "together/t5-11b":
            register_window_service_for_model(model_name, T511bWindowService)
        elif model_name == "together/ul2":
            register_window_service_for_model(model_name, UL2WindowService)
        elif model_name == "together/yalm":
            register_window_service_for_model(model_name, YaLMWindowService)
        elif organization == "cohere":
            register_window_service_for_model(model_name, CohereWindowService)
        elif organization == "ai21":
            register_window_service_for_model(
                model_name,
                lambda service: AI21WindowService(service=service, gpt2_window_service=GPT2WindowService(service)),
            )
        else:
            raise ValueError(f"Unhandled model name: {model_name}")


_register_build_in_window_services()


class WindowServiceFactory:
    @staticmethod
    def get_window_service(model_name: str, service: TokenizerService) -> WindowService:
        """
        Returns a `WindowService` given the name of the model.
        Make sure this function returns instantaneously on repeated calls.
        """
        window_service_factory = _window_service_factory_registry.get(model_name)
        if window_service_factory is None:
            raise ValueError(
                f"No registered window service found for model: {model_name} - "
                "register a window service using register_window_service_for_model()"
            )
        return window_service_factory(service)
