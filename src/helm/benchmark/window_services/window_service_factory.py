from helm.benchmark.model_deployment_registry import WindowServiceSpec, get_model_deployment
from helm.benchmark.window_services.gpt2_window_service import GPT2WindowService
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

from helm.benchmark.window_services.default_window_service import DefaultWindowService
from helm.benchmark.window_services.huggingface_window_service import HuggingFaceWindowService
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
            window_service = DefaultWindowService(
                service,
                tokenizer_name="neurips/local",
                max_sequence_length=2048,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
        elif organization == "openai":
            if model_name in get_model_names_with_tag(GPT4_CONTEXT_WINDOW_TAG):
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="openai/cl100k_base",
                    max_sequence_length=8192,
                    max_request_length=8193,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
            elif model_name in get_model_names_with_tag(GPT4_32K_CONTEXT_WINDOW_TAG):
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="openai/cl100k_base",
                    max_sequence_length=32768,
                    max_request_length=32769,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
            elif model_name in get_model_names_with_tag(GPT_TURBO_CONTEXT_WINDOW_TAG):
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="openai/cl100k_base",
                    max_sequence_length=4000,
                    max_request_length=4001,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
            elif model_name in get_model_names_with_tag(GPT_TURBO_16K_CONTEXT_WINDOW_TAG):
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="openai/cl100k_base",
                    max_sequence_length=16000,
                    max_request_length=16001,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
            elif model_name in get_model_names_with_tag(WIDER_CONTEXT_WINDOW_TAG):
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="huggingface/gpt2",
                    max_sequence_length=4000,
                    max_request_length=4001,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
            else:
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="huggingface/gpt2",
                    max_sequence_length=2048,
                    max_request_length=2049,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
        # For the Google models, we approximate with the OpenAIWindowService
        elif organization == "simple" or organization == "google":
            window_service = DefaultWindowService(
                service,
                tokenizer_name="huggingface/gpt2",
                max_sequence_length=2048,
                max_request_length=2049,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
        elif organization == "AlephAlpha":
            # From https://docs.aleph-alpha.com/api/complete, "the summed number of tokens of prompt
            # and maximum_tokens may not exceed 2048 tokens." Confirmed it's 2048 for the Luminous
            # models currently available.
            if engine == "luminous-base":
                window_service = DefaultWindowService(
                    service, tokenizer_name="AlephAlpha/luminous-base", max_sequence_length=2048
                )
            elif engine == "luminous-extended":
                window_service = DefaultWindowService(
                    service, tokenizer_name="AlephAlpha/luminous-extended", max_sequence_length=2048
                )
            elif engine == "luminous-supreme":
                window_service = DefaultWindowService(
                    service, tokenizer_name="AlephAlpha/luminous-supreme", max_sequence_length=2048
                )
            elif engine == "luminous-world":
                window_service = DefaultWindowService(
                    service, tokenizer_name="AlephAlpha/luminous-world", max_sequence_length=2048
                )
            else:
                raise ValueError(f"Unhandled Aleph Alpha model: {engine}")
        elif organization == "microsoft":
            # The max request length for the MT-NLG models is 2048.
            # Source: https://github.com/microsoft/turing-academic-TNLG
            #
            # MT-NLG does not predict the logprob of the first
            # input token so `max_sequence_length` is one token shorter than `max_request_length`.
            window_service = DefaultWindowService(
                service,
                tokenizer_name="huggingface/gpt2",
                max_sequence_length=2047,
                max_request_length=2048,
                end_of_text_token="<|endoftext|>",
                prefix_token="<<",
            )
        elif organization == "anthropic":
            if engine == "stanford-online-all-v4-s3":
                # For the legacy Anthropic mode, the tokenizer was not publicly available,
                # so approximate with the GPT-2 tokenizer.
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="huggingface/gpt2",
                    max_sequence_length=8192,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
            else:
                # While the max_sequence_length limit seems to be 8192, we limit to 8000
                # according to Anthropic's recommendations.
                # See: https://console.anthropic.com/docs/prompt-design
                #
                # Claude is one of the rare models that has a limit on max_sequence_and_generated_tokens_length.
                # The official limit seems to be 9192,but using scripts/compute_request_limits.py
                # we found that the limit is actually 9016.
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="anthropic/claude",
                    max_sequence_length=8000,
                    max_sequence_and_generated_tokens_length=9016,
                    end_of_text_token="<|endoftext|>",
                    prefix_token="<|endoftext|>",
                )
        elif organization == "writer":
            if engine in ["palmyra-base", "palmyra-large", "palmyra-instruct-30", "palmyra-e"]:
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="huggingface/gpt2",
                    max_sequence_length=2048,
                    max_sequence_and_generated_tokens_length=2048,
                )
            elif engine in ["palmyra-x", "silk-road"]:
                window_service = DefaultWindowService(
                    service,
                    tokenizer_name="huggingface/gpt2",
                    max_sequence_length=8192,
                    max_sequence_and_generated_tokens_length=8192,
                )
            else:
                raise ValueError(f"Unhandled Writer model: {engine}")
        elif engine == "santacoder":
            window_service = DefaultWindowService(
                service,
                tokenizer_name="bigcode/santacoder",
                max_sequence_length=2048,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
        elif engine == "starcoder":
            window_service = DefaultWindowService(
                service,
                tokenizer_name="bigcode/starcoder",
                max_sequence_length=8192,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
        elif model_name in ["huggingface/gpt2", "together/h3-2.7b"]:
            window_service = GPT2WindowService(service)
        elif model_name == "together/bloom":
            # Source: https://huggingface.co/bigscience/bloom
            window_service = DefaultWindowService(
                service,
                tokenizer_name="bigscience/bloom",
                max_sequence_length=2048,
                max_request_length=2049,
                end_of_text_token="</s>",
                prefix_token="</s>",
            )
        elif model_name == "together/glm":
            # From https://github.com/THUDM/GLM-130B, "the tokenizer is implemented based on
            # icetk---a unified multimodal tokenizer for images, Chinese, and English."
            from helm.benchmark.window_services.ice_window_service import ICEWindowService

            window_service = ICEWindowService(service)
        elif model_name in ["huggingface/gpt-j-6b", "together/gpt-j-6b", "together/gpt-jt-6b-v1", "gooseai/gpt-j-6b"]:
            # The same tokenizer as GPT-2, but with an additional 143 tokens
            # (source: https://huggingface.co/docs/transformers/model_doc/gptj).
            window_service = DefaultWindowService(
                service,
                tokenizer_name="EleutherAI/gpt-j-6B",
                max_sequence_length=2048,
                max_request_length=2049,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
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
            window_service = DefaultWindowService(
                service,
                tokenizer_name="EleutherAI/gpt-neox-20b",
                max_sequence_length=2048,
                max_request_length=2049,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
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
            # The context length for these models is 4096 tokens.
            # See: https://github.com/Stability-AI/StableLM#stablelm-alpha
            window_service = DefaultWindowService(
                service,
                tokenizer_name="EleutherAI/gpt-neox-20b",
                max_sequence_length=4096,
                max_request_length=4097,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
        elif model_name in [
            "together/opt-1.3b",
            "together/opt-6.7b",
            "together/opt-66b",
            "together/opt-175b",
        ]:
            # The max sequence length for the OPT models is 2048.
            # Source: https://arxiv.org/pdf/2205.01068.pdf
            window_service = DefaultWindowService(
                service,
                tokenizer_name="facebook/opt-66b",
                max_sequence_length=2048,
                max_request_length=2049,
                end_of_text_token="</s>",
                prefix_token="</s>",
            )
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
            # The only difference between this and GPT2WindowService is that
            # the request length is constrained to the sequence length.
            window_service = DefaultWindowService(
                service,
                tokenizer_name="huggingface/gpt2",
                max_sequence_length=1024,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
        elif model_name in [
            "lmsys/vicuna-7b-v1.3",
            "lmsys/vicuna-13b-v1.3",
            "meta/llama-7b",
            "meta/llama-13b",
            "meta/llama-30b",
            "meta/llama-65b",
            "stanford/alpaca-7b",
        ]:
            # Tokenizer name hf-internal-testing/llama-tokenizer is taken from:
            # https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaTokenizerFast.example
            window_service = HuggingFaceWindowService(service, tokenizer_name="hf-internal-testing/llama-tokenizer")
        elif model_name in [
            "meta/llama-2-7b",
            "meta/llama-2-13b",
            "meta/llama-2-70b",
        ]:
            # To use the Llama-2 tokenizer:
            #
            # 1. Accept the license agreement: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
            # 2. Request to access the Hugging Face repository: https://huggingface.co/meta-llama/Llama-2-7b
            # 3. Run `huggingface-cli login`
            #
            # If you encounter the following error, complete the above steps and try again:
            #
            #     meta-llama/Llama-2-70b-hf is not a local folder and is not a valid model identifier listed on
            #     'https://huggingface.co/models'

            window_service = HuggingFaceWindowService(
                service, tokenizer_name="meta-llama/Llama-2-7b-hf", max_sequence_length=4096, max_request_length=4096
            )
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
            window_service = DefaultWindowService(
                service,
                tokenizer_name="lightningai/lit-gpt",
                max_sequence_length=2048,
                end_of_text_token="<|endoftext|>",
                prefix_token="<|endoftext|>",
            )
        elif organization == "mistralai":
            window_service = HuggingFaceWindowService(service, tokenizer_name="mistralai/Mistral-7B-v0.1")
        elif model_name in [
            "HuggingFaceM4/idefics-9b",
            "HuggingFaceM4/idefics-9b-instruct",
            "HuggingFaceM4/idefics-80b",
            "HuggingFaceM4/idefics-80b-instruct",
        ]:
            window_service = HuggingFaceWindowService(service, tokenizer_name=model_name)
        else:
            raise ValueError(f"Unhandled model name: {model_name}")

        return window_service
