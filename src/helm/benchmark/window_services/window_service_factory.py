from helm.benchmark.model_deployment_registry import get_model_deployment
from helm.benchmark.tokenizer_config_registry import get_tokenizer_config
from helm.proxy.clients.huggingface_model_registry import HuggingFaceHubModelConfig
from helm.proxy.models import (
    get_model,
    get_model_names_with_tag,
    Model,
    AI21_WIDER_CONTEXT_WINDOW_TAG,
    AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG,
    WIDER_CONTEXT_WINDOW_TAG,
    GPT_TURBO_CONTEXT_WINDOW_TAG,
    GPT_TURBO_16K_CONTEXT_WINDOW_TAG,
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
    GPTTurboWindowService,
    GPTTurbo16KWindowService,
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
from .gptneox_window_service import GPTNeoXWindowService, StableLMAlphaWindowService
from .megatron_window_service import MegatronWindowService
from .opt_window_service import OPTWindowService
from .palmyra_window_service import PalmyraWindowService, LongerPalmyraWindowService
from .remote_window_service import get_remote_window_service
from .t0pp_window_service import T0ppWindowService
from .t511b_window_service import T511bWindowService
from .flan_t5_window_service import FlanT5WindowService
from .ul2_window_service import UL2WindowService
from .yalm_window_service import YaLMWindowService
from .llama_window_service import LlamaWindowService, Llama2WindowService
from .window_service import WindowService
from .tokenizer_service import TokenizerService
from .http_model_window_service import HTTPModelWindowServce
from helm.benchmark.window_services.configurable_tokenizer_window_service import ConfigurableTokenizerWindowService
from helm.proxy.clients.huggingface_client import get_huggingface_model_config
from helm.proxy.clients.remote_model_registry import get_remote_model


class WindowServiceFactory:
    @staticmethod
    def get_window_service(model_name: str, service: TokenizerService) -> WindowService:
        """
        Returns a `WindowService` given the name of the model.
        Make sure this function returns instantaneously on repeated calls.
        """

        # Catch any HuggingFace models registered via the command line flags
        huggingface_model_config = get_huggingface_model_config(model_name)

        # TODO: Migrate all window services to use use model deployments
        model_deployment = get_model_deployment(model_name)
        print(f"[debug:yifanmai] Gotten? model_deployment {model_deployment}")
        if model_deployment:
            # TODO: Allow tokenizer name auto-inference in some cases.
            if not model_deployment.tokenizer_name:
                raise Exception("Tokenizer name must be set on model deplyment")
            tokenizer_name = model_deployment.tokenizer_name
            tokenizer_config = get_tokenizer_config(tokenizer_name)
            if tokenizer_config:
                return ConfigurableTokenizerWindowService(service, tokenizer_config, model_deployment)
            else:
                # Fall back to HuggingFaceWindowService.
                # This auto-infers the tokenizer's properties (e.g. special tokens)
                # using Hugging Face Hub.
                return HuggingFaceWindowService(
                    service=service,
                    model_config=HuggingFaceHubModelConfig.from_string(tokenizer_name),
                    max_sequence_length=model_deployment.max_sequence_length,
                )
        elif get_remote_model(model_name):
            window_service = get_remote_window_service(service, model_name)
        elif organization == "neurips":
            window_service = HTTPModelWindowServce(service)
        elif huggingface_model_config:
            return HuggingFaceWindowService(service=service, model_config=huggingface_model_config)
        
        model: Model = get_model(model_name)
        organization: str = model.organization
        engine: str = model.engine
        if organization == "openai":
            if model_name in get_model_names_with_tag(GPT4_CONTEXT_WINDOW_TAG):
                return GPT4WindowService(service)
            elif model_name in get_model_names_with_tag(GPT4_32K_CONTEXT_WINDOW_TAG):
                return GPT432KWindowService(service)
            if model_name in get_model_names_with_tag(GPT_TURBO_CONTEXT_WINDOW_TAG):
                return GPTTurboWindowService(service)
            elif model_name in get_model_names_with_tag(GPT_TURBO_16K_CONTEXT_WINDOW_TAG):
                return GPTTurbo16KWindowService(service)
            elif model_name in get_model_names_with_tag(WIDER_CONTEXT_WINDOW_TAG):
                return WiderOpenAIWindowService(service)
            else:
                return OpenAIWindowService(service)
        # For the Google models, we approximate with the OpenAIWindowService
        elif organization == "simple" or organization == "google":
            return OpenAIWindowService(service)
        elif organization == "AlephAlpha":
            if engine == "luminous-base":
                return LuminousBaseWindowService(service)
            elif engine == "luminous-extended":
                return LuminousExtendedWindowService(service)
            elif engine == "luminous-supreme":
                return LuminousSupremeWindowService(service)
            elif engine == "luminous-world":
                return LuminousWorldWindowService(service)
            else:
                raise ValueError(f"Unhandled Aleph Alpha model: {engine}")
        elif organization == "microsoft":
            return MTNLGWindowService(service)
        elif organization == "anthropic":
            if engine == "stanford-online-all-v4-s3":
                return LegacyAnthropicWindowService(service)
            else:
                return AnthropicWindowService(service)
        elif organization == "writer":
            if engine in ["palmyra-base", "palmyra-large", "palmyra-instruct-30", "palmyra-e"]:
                return PalmyraWindowService(service)
            elif engine in ["palmyra-x", "silk-road"]:
                return LongerPalmyraWindowService(service)
            else:
                raise ValueError(f"Unhandled Writer model: {engine}")
        elif engine == "santacoder":
            return SantaCoderWindowService(service)
        elif engine == "starcoder":
            return StarCoderWindowService(service)
        elif model_name == "huggingface/gpt2":
            return GPT2WindowService(service)
        elif model_name == "together/bloom":
            return BloomWindowService(service)
        elif model_name == "together/glm":
            # From https://github.com/THUDM/GLM-130B, "the tokenizer is implemented based on
            # icetk---a unified multimodal tokenizer for images, Chinese, and English."
            return ICEWindowService(service)
        elif model_name in ["huggingface/gpt-j-6b", "together/gpt-j-6b", "together/gpt-jt-6b-v1", "gooseai/gpt-j-6b"]:
            return GPTJWindowService(service)
        elif model_name in [
            "together/gpt-neox-20b",
            "gooseai/gpt-neo-20b",
            "together/gpt-neoxt-chat-base-20b",
            "together/redpajama-incite-base-3b-v1",
            "together/redpajama-incite-instruct-3b-v1",
            "together/redpajama-incite-base-7b",
            "together/redpajama-incite-instruct-7b",
            # Pythia uses the same tokenizer as GPT-NeoX-20B.
            # See: https://huggingface.co/EleutherAI/pythia-6.9b#training-procedure
            "eleutherai/pythia-1b-v0",
            "eleutherai/pythia-2.8b-v0",
            "eleutherai/pythia-6.9b",
            "eleutherai/pythia-12b-v0",
            # MPT-7B model was trained with the EleutherAI/gpt-neox-20b tokenizer
            # See: https://huggingface.co/mosaicml/mpt-7b
            "mosaicml/mpt-7b",
            # Dolly models are based on Pythia.
            # See: https://github.com/databrickslabs/dolly
            "databricks/dolly-v2-3b",
            "databricks/dolly-v2-7b",
            "databricks/dolly-v2-12b",
        ]:
            return GPTNeoXWindowService(service)
        elif model_name in [
            "stabilityai/stablelm-base-alpha-3b",
            "stabilityai/stablelm-base-alpha-7b",
        ]:
            return StableLMAlphaWindowService(service)
        elif model_name == "together/h3-2.7b":
            return GPT2WindowService(service)
        elif model_name in [
            "together/opt-1.3b",
            "together/opt-6.7b",
            "together/opt-66b",
            "together/opt-175b",
        ]:
            return OPTWindowService(service)
        elif model_name == "together/t0pp":
            return T0ppWindowService(service)
        elif model_name == "together/t5-11b":
            return T511bWindowService(service)
        elif model_name == "together/flan-t5-xxl":
            return FlanT5WindowService(service)
        elif model_name == "together/ul2":
            return UL2WindowService(service)
        elif model_name == "together/yalm":
            return YaLMWindowService(service)
        elif model_name == "nvidia/megatron-gpt2":
            return MegatronWindowService(service)
        elif model_name in [
            "meta/llama-7b",
            "meta/llama-13b",
            "meta/llama-30b",
            "meta/llama-65b",
            "together/alpaca-7b",
            "together/vicuna-13b",
        ]:
            return LlamaWindowService(service)
        elif model_name in [
            "meta/llama-2-7b",
            "meta/llama-2-13b",
            "meta/llama-2-70b",
        ]:
            return Llama2WindowService(service)
        elif organization == "cohere":
            if "command" in engine:
                return CohereCommandWindowService(service)
            else:
                return CohereWindowService(service)
        elif organization == "ai21":
            if model_name in get_model_names_with_tag(AI21_WIDER_CONTEXT_WINDOW_TAG):
                return WiderAI21WindowService(service=service, gpt2_window_service=GPT2WindowService(service))
            if model_name in get_model_names_with_tag(AI21_JURASSIC_2_JUMBO_CONTEXT_WINDOW_TAG):
                return AI21Jurassic2JumboWindowService(
                    service=service, gpt2_window_service=GPT2WindowService(service)
                )
            else:
                return AI21WindowService(service=service, gpt2_window_service=GPT2WindowService(service))
        else:
            raise ValueError(f"Unhandled model name: {model_name}")
