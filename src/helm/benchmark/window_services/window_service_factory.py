from helm.benchmark.model_deployment_registry import WindowServiceSpec, get_model_deployment
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
from helm.proxy.clients.remote_model_registry import get_remote_model
from helm.common.object_spec import create_object, inject_object_spec_args


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

        # TODO: Migrate all window services to use use model deployments
        model_deployment = get_model_deployment(model_name)
        if model_deployment:
            # If the model deployment specifies a WindowServiceSpec, instantiate it.
            window_service_spec: WindowServiceSpec
            if model_deployment.window_service_spec:
                window_service_spec = model_deployment.window_service_spec
            else:
                window_service_spec = WindowServiceSpec(
                    class_name="helm.benchmark.window_services.default_window_service.DefaultWindowService", args={}
                )
            # Perform dependency injection to fill in remaining arguments.
            # Dependency injection is needed here for these reasons:
            #
            # 1. Different window services have different parameters. Dependency injection provides arguments
            #    that match the parameters of the window services.
            # 2. Some arguments, such as the tokenizer service, are not static data objects that can be
            #    in the users configuration file. Instead, they have to be constructed dynamically at runtime.
            window_service_spec = inject_object_spec_args(
                window_service_spec,
                {
                    "service": service,
                    "tokenizer_name": model_deployment.tokenizer_name,
                    "max_sequence_length": model_deployment.max_sequence_length,
                    "max_request_length": model_deployment.max_request_length,
                },
            )
            window_service = create_object(window_service_spec)
        elif get_remote_model(model_name):
            window_service = get_remote_window_service(service, model_name)
        elif organization == "neurips":
            from helm.benchmark.window_services.http_model_window_service import HTTPModelWindowServce

            window_service = HTTPModelWindowServce(service)
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
            window_service = HuggingFaceWindowService(service=service, tokenizer_name="tiiuae/falcon-7b")
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

        elif organization == "lightningai":
            from helm.benchmark.window_services.lit_gpt_window_service import LitGPTWindowServce

            window_service = LitGPTWindowServce(service)
        elif organization == "mistralai":
            window_service = HuggingFaceWindowService(service, tokenizer_name="mistralai/Mistral-7B-v0.1")
        elif model_name in [
            "HuggingFaceM4/idefics-9b",
            "HuggingFaceM4/idefics-9b-instruct",
            "HuggingFaceM4/idefics-80b",
            "HuggingFaceM4/idefics-80b-instruct",
        ]:
            window_service = HuggingFaceWindowService(service, model_name)
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        return window_service
