from helm.benchmark.model_deployment_registry import get_model_deployment
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

from helm.benchmark.window_services.huggingface_window_service import HuggingFaceWindowService
from helm.benchmark.window_services.gpt2_window_service import GPT2WindowService
from helm.benchmark.window_services.remote_window_service import get_remote_window_service
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.tokenizer_service import TokenizerService
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
        # Catch any HuggingFace models registered via the command line flags
        huggingface_model_config = get_huggingface_model_config(model_name)

        # TODO: Migrate all window services to use use model deployments
        model_deployment = get_model_deployment(model_name)
        if model_deployment:
            # TODO: Allow tokenizer name auto-inference in some cases.
            if not model_deployment.tokenizer_name:
                raise Exception("Tokenizer name must be set on model deplyment")
            tokenizer_name = model_deployment.tokenizer_name
            # Only use HuggingFaceWindowService for now.
            # TODO: Allow using other window services.
            window_service = HuggingFaceWindowService(
                service=service,
                model_config=HuggingFaceHubModelConfig.from_string(tokenizer_name),
                max_sequence_length=model_deployment.max_sequence_length,
            )
        elif get_remote_model(model_name):
            window_service = get_remote_window_service(service, model_name)
        elif organization == "neurips":
            from helm.benchmark.window_services.http_model_window_service import HTTPModelWindowServce

            window_service = HTTPModelWindowServce(service)
        elif huggingface_model_config:
            window_service = HuggingFaceWindowService(service=service, model_config=huggingface_model_config)
        elif organization == "openai":
            from helm.benchmark.window_services.openai_window_service import OpenAIWindowService
            from helm.benchmark.window_services.wider_openai_window_service import (
                WiderOpenAIWindowService,
                GPTTurboWindowService,
                GPTTurbo16KWindowService,
                GPT4WindowService,
                GPT432KWindowService,
            )

            if model_name in get_model_names_with_tag(GPT4_CONTEXT_WINDOW_TAG):
                window_service = GPT4WindowService(service)
            elif model_name in get_model_names_with_tag(GPT4_32K_CONTEXT_WINDOW_TAG):
                window_service = GPT432KWindowService(service)
            if model_name in get_model_names_with_tag(GPT_TURBO_CONTEXT_WINDOW_TAG):
                window_service = GPTTurboWindowService(service)
            elif model_name in get_model_names_with_tag(GPT_TURBO_16K_CONTEXT_WINDOW_TAG):
                window_service = GPTTurbo16KWindowService(service)
            elif model_name in get_model_names_with_tag(WIDER_CONTEXT_WINDOW_TAG):
                window_service = WiderOpenAIWindowService(service)
            else:
                window_service = OpenAIWindowService(service)
        # For the Google models, we approximate with the OpenAIWindowService
        elif organization == "simple" or organization == "google":
            from helm.benchmark.window_services.openai_window_service import OpenAIWindowService

            window_service = OpenAIWindowService(service)
        elif organization == "AlephAlpha":
            from helm.benchmark.window_services.luminous_window_service import (
                LuminousBaseWindowService,
                LuminousExtendedWindowService,
                LuminousSupremeWindowService,
                LuminousWorldWindowService,
            )

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
            from helm.benchmark.window_services.mt_nlg_window_service import MTNLGWindowService

            window_service = MTNLGWindowService(service)
        elif organization == "anthropic":
            from helm.benchmark.window_services.anthropic_window_service import (
                AnthropicWindowService,
                LegacyAnthropicWindowService,
            )

            if engine == "stanford-online-all-v4-s3":
                window_service = LegacyAnthropicWindowService(service)
            else:
                window_service = AnthropicWindowService(service)
        elif organization == "writer":
            from helm.benchmark.window_services.palmyra_window_service import (
                PalmyraWindowService,
                LongerPalmyraWindowService,
            )

            if engine in ["palmyra-base", "palmyra-large", "palmyra-instruct-30", "palmyra-e"]:
                window_service = PalmyraWindowService(service)
            elif engine in ["palmyra-x", "silk-road"]:
                window_service = LongerPalmyraWindowService(service)
            else:
                raise ValueError(f"Unhandled Writer model: {engine}")
        elif engine == "santacoder":
            from helm.benchmark.window_services.santacoder_window_service import SantaCoderWindowService

            window_service = SantaCoderWindowService(service)
        elif engine == "starcoder":
            from helm.benchmark.window_services.starcoder_window_service import StarCoderWindowService

            window_service = StarCoderWindowService(service)
        elif model_name == "huggingface/gpt2":
            window_service = GPT2WindowService(service)
        elif model_name == "together/bloom":
            from helm.benchmark.window_services.bloom_window_service import BloomWindowService

            window_service = BloomWindowService(service)
        elif model_name == "together/glm":
            # From https://github.com/THUDM/GLM-130B, "the tokenizer is implemented based on
            # icetk---a unified multimodal tokenizer for images, Chinese, and English."
            from helm.benchmark.window_services.ice_window_service import ICEWindowService

            window_service = ICEWindowService(service)
        elif model_name in ["huggingface/gpt-j-6b", "together/gpt-j-6b", "together/gpt-jt-6b-v1", "gooseai/gpt-j-6b"]:
            from helm.benchmark.window_services.gptj_window_service import GPTJWindowService

            window_service = GPTJWindowService(service)
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
            "mosaicml/mpt-instruct-7b",
            "mosaicml/mpt-30b",
            "mosaicml/mpt-instruct-30b",
            # Dolly models are based on Pythia.
            # See: https://github.com/databrickslabs/dolly
            "databricks/dolly-v2-3b",
            "databricks/dolly-v2-7b",
            "databricks/dolly-v2-12b",
        ]:
            from helm.benchmark.window_services.gptneox_window_service import GPTNeoXWindowService

            window_service = GPTNeoXWindowService(service)
        elif model_name in [
            "tiiuae/falcon-7b",
            "tiiuae/falcon-7b-instruct",
            "tiiuae/falcon-40b",
            "tiiuae/falcon-40b-instruct",
        ]:
            window_service = HuggingFaceWindowService(
                service=service,
                model_config=HuggingFaceHubModelConfig(namespace="tiiuae", model_name="falcon-7b", revision=None),
            )
        elif model_name in [
            "stabilityai/stablelm-base-alpha-3b",
            "stabilityai/stablelm-base-alpha-7b",
        ]:
            from helm.benchmark.window_services.gptneox_window_service import StableLMAlphaWindowService

            window_service = StableLMAlphaWindowService(service)
        elif model_name == "together/h3-2.7b":
            window_service = GPT2WindowService(service)
        elif model_name in [
            "together/opt-1.3b",
            "together/opt-6.7b",
            "together/opt-66b",
            "together/opt-175b",
        ]:
            from helm.benchmark.window_services.opt_window_service import OPTWindowService

            window_service = OPTWindowService(service)
        elif model_name == "together/t0pp":
            from helm.benchmark.window_services.t0pp_window_service import T0ppWindowService

            window_service = T0ppWindowService(service)
        elif model_name == "together/t5-11b":
            from helm.benchmark.window_services.t511b_window_service import T511bWindowService

            window_service = T511bWindowService(service)
        elif model_name == "together/flan-t5-xxl":
            from helm.benchmark.window_services.flan_t5_window_service import FlanT5WindowService

            window_service = FlanT5WindowService(service)
        elif model_name == "together/ul2":
            from helm.benchmark.window_services.ul2_window_service import UL2WindowService

            window_service = UL2WindowService(service)
        elif model_name == "together/yalm":
            from helm.benchmark.window_services.yalm_window_service import YaLMWindowService

            window_service = YaLMWindowService(service)
        elif model_name == "nvidia/megatron-gpt2":
            from helm.benchmark.window_services.megatron_window_service import MegatronWindowService

            window_service = MegatronWindowService(service)
        elif model_name in [
            "lmsys/vicuna-7b-v1.3",
            "lmsys/vicuna-13b-v1.3",
            "meta/llama-7b",
            "meta/llama-13b",
            "meta/llama-30b",
            "meta/llama-65b",
            "stanford/alpaca-7b",
        ]:
            from helm.benchmark.window_services.llama_window_service import LlamaWindowService

            window_service = LlamaWindowService(service)
        elif model_name in [
            "meta/llama-2-7b",
            "meta/llama-2-13b",
            "meta/llama-2-70b",
        ]:
            from helm.benchmark.window_services.llama_window_service import Llama2WindowService

            window_service = Llama2WindowService(service)
        elif organization == "cohere":
            from helm.benchmark.window_services.cohere_window_service import (
                CohereWindowService,
                CohereCommandWindowService,
            )

            if "command" in engine:
                window_service = CohereCommandWindowService(service)
            else:
                window_service = CohereWindowService(service)
        elif organization == "ai21":
            from helm.benchmark.window_services.wider_ai21_window_service import (
                WiderAI21WindowService,
                AI21Jurassic2JumboWindowService,
            )
            from helm.benchmark.window_services.ai21_window_service import AI21WindowService

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
