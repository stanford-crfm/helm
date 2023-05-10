from helm.proxy.models import (
    get_model,
    get_model_names_with_tag,
    Model,
    AI21_WIDER_CONTEXT_WINDOW_TAG,
    AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG,
    WIDER_CONTEXT_WINDOW_TAG,
    GPT4_TOKENIZER_TAG,
    GPT4_CONTEXT_WINDOW_TAG,
    GPT4_32K_CONTEXT_WINDOW_TAG,
)
from .ai21_window_service import AI21WindowService
from .wider_ai21_window_service import WiderAI21WindowService, AI21Jurassic2JumboWindowService
from .anthropic_window_service import AnthropicWindowService, LegacyAnthropicWindowService
from .cohere_window_service import CohereWindowService, CohereCommandWindowService
from .luminous_window_service import (
    LuminousBaseWindowService,
    LuminousExtendedWindowService,
    LuminousSupremeWindowService,
    LuminousWorldWindowService,
)
from .openai_window_service import OpenAIWindowService
from .wider_openai_window_service import (
    WiderOpenAIWindowService,
    GPT3Point5TurboWindowService,
    GPT4WindowService,
    GPT432KWindowService,
)
from .mt_nlg_window_service import MTNLGWindowService
from .bloom_window_service import BloomWindowService
from .huggingface_window_service import HuggingFaceWindowService
from .ice_window_service import ICEWindowService
from .santacoder_window_service import SantaCoderWindowService
from .starcoder_window_service import StarCoderWindowService
from .gpt2_window_service import GPT2WindowService
from .gptj_window_service import GPTJWindowService
from .gptneox_window_service import GPTNeoXWindowService
from .megatron_window_service import MegatronWindowService
from .opt_window_service import OPTWindowService
from .palmyra_window_service import PalmyraWindowService, SilkRoadWindowService
from .remote_window_service import get_remote_window_service
from .t0pp_window_service import T0ppWindowService
from .t511b_window_service import T511bWindowService
from .flan_t5_window_service import FlanT5WindowService
from .ul2_window_service import UL2WindowService
from .yalm_window_service import YaLMWindowService
from .window_service import WindowService
from .tokenizer_service import TokenizerService
from helm.proxy.clients.huggingface_client import get_huggingface_model_config
from helm.proxy.clients.remote_model_registry import get_remote_model


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
        huggingface_model_config = get_huggingface_model_config(model_name)
        if get_remote_model(model_name):
            window_service = get_remote_window_service(service, model_name)
        elif huggingface_model_config:
            window_service = HuggingFaceWindowService(service=service, model_config=huggingface_model_config)
        elif organization == "openai":
            if model_name in get_model_names_with_tag(GPT4_CONTEXT_WINDOW_TAG):
                window_service = GPT4WindowService(service)
            elif model_name in get_model_names_with_tag(GPT4_32K_CONTEXT_WINDOW_TAG):
                window_service = GPT432KWindowService(service)
            elif model_name in get_model_names_with_tag(GPT4_TOKENIZER_TAG):
                window_service = GPT3Point5TurboWindowService(service)
            elif model_name in get_model_names_with_tag(WIDER_CONTEXT_WINDOW_TAG):
                window_service = WiderOpenAIWindowService(service)
            else:
                window_service = OpenAIWindowService(service)
        # For the Google models, we approximate with the OpenAIWindowService
        elif organization == "simple" or organization == "google":
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
            if engine == "stanford-online-all-v4-s3":
                window_service = LegacyAnthropicWindowService(service)
            else:
                window_service = AnthropicWindowService(service)
        elif organization == "writer":
            if engine in ["palmyra-base", "palmyra-large", "palmyra-instruct-30", "palmyra-e"]:
                window_service = PalmyraWindowService(service)
            elif engine == "silk-road":
                window_service = SilkRoadWindowService(service)
            else:
                raise ValueError(f"Unhandled Writer model: {engine}")
        elif engine == "santacoder":
            window_service = SantaCoderWindowService(service)
        elif engine == "starcoder":
            window_service = StarCoderWindowService(service)
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
        elif model_name in ["together/gpt-neox-20b", "gooseai/gpt-neo-20b", "together/gpt-neoxt-chat-base-20b"]:
            window_service = GPTNeoXWindowService(service)
        elif model_name == "together/h3-2.7b":
            window_service = GPT2WindowService(service)
        elif model_name in ["together/opt-1.3b", "together/opt-6.7b", "together/opt-66b", "together/opt-175b"]:
            window_service = OPTWindowService(service)
        elif model_name == "together/t0pp":
            window_service = T0ppWindowService(service)
        elif model_name == "together/t5-11b":
            window_service = T511bWindowService(service)
        elif model_name == "together/flan-t5-xxl":
            window_service = FlanT5WindowService(service)
        elif model_name == "together/ul2":
            window_service = UL2WindowService(service)
        elif model_name == "together/yalm":
            window_service = YaLMWindowService(service)
        elif model_name == "nvidia/megatron-gpt2":
            window_service = MegatronWindowService(service)
        elif organization == "cohere":
            if "command" in engine:
                window_service = CohereCommandWindowService(service)
            else:
                window_service = CohereWindowService(service)
        elif organization == "ai21":
            if model_name in get_model_names_with_tag(AI21_WIDER_CONTEXT_WINDOW_TAG):
                window_service = WiderAI21WindowService(service=service, gpt2_window_service=GPT2WindowService(service))
            if model_name in get_model_names_with_tag(AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG):
                window_service = AI21Jurassic2JumboWindowService(
                    service=service, gpt2_window_service=GPT2WindowService(service)
                )
            else:
                window_service = AI21WindowService(service=service, gpt2_window_service=GPT2WindowService(service))
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        return window_service
