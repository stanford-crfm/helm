from typing import Dict, Callable

from proxy.models import get_model, Model, WIDER_CONTEXT_WINDOW_TAG
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


# Register built-in window service factories
register_window_service_for_model("huggingface/gpt2", GPT2WindowService)
register_window_service_for_model("together/bloom", BloomWindowService)
register_window_service_for_model(
    "together/glm",
    # From https://github.com/THUDM/GLM-130B, "the tokenizer is implemented based on
    # icetk---a unified multimodal tokenizer for images, Chinese, and English."
    ICEWindowService,
)
register_window_service_for_model("huggingface/gpt-j-6b", GPTJWindowService)
register_window_service_for_model("together/gpt-j-6b", GPTJWindowService)
register_window_service_for_model("gooseai/gpt-j-6b", GPTJWindowService)
register_window_service_for_model("together/gpt-neox-20b", GPTNeoXWindowService)
register_window_service_for_model("gooseai/gpt-neo-20b", GPTNeoXWindowService)
register_window_service_for_model("together/opt-66b", OPTWindowService)
register_window_service_for_model("together/opt-175b", OPTWindowService)
register_window_service_for_model("together/t0pp", T0ppWindowService)
register_window_service_for_model("together/t5-11b", T511bWindowService)
register_window_service_for_model("together/ul2", UL2WindowService)
register_window_service_for_model("together/yalm", YaLMWindowService)


class WindowServiceFactory:
    @staticmethod
    def get_window_service(model_name: str, service: TokenizerService) -> WindowService:
        """
        Returns a `WindowService` given the name of the model.
        Make sure this function returns instantaneously on repeated calls.
        """
        wondow_service_factory_for_model = _window_service_factory_registry.get(model_name)
        if wondow_service_factory_for_model is not None:
            return wondow_service_factory_for_model(service)

        model: Model = get_model(model_name)
        organization: str = model.organization

        window_service: WindowService
        if WIDER_CONTEXT_WINDOW_TAG in model.tags:
            window_service = WiderOpenAIWindowService(service)
        elif organization == "openai" or organization == "simple":
            window_service = OpenAIWindowService(service)
        elif organization == "microsoft":
            window_service = MTNLGWindowService(service)
        elif organization == "anthropic":
            window_service = AnthropicWindowService(service)
        elif organization == "cohere":
            window_service = CohereWindowService(service)
        elif organization == "ai21":
            window_service = AI21WindowService(service=service, gpt2_window_service=GPT2WindowService(service))
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        return window_service
