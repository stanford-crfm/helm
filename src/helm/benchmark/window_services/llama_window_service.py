from helm.proxy.clients.huggingface_client import HuggingFaceModelConfig
from helm.benchmark.window_services.huggingface_window_service import HuggingFaceWindowService
from helm.benchmark.window_services.tokenizer_service import TokenizerService


class LlamaWindowService(HuggingFaceWindowService):
    def __init__(self, service: TokenizerService):
        # Tokenizer name hf-internal-testing/llama-tokenizer is taken from:
        # https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaTokenizerFast.example
        model_config = HuggingFaceModelConfig(
            namespace="hf-internal-testing", model_name="llama-tokenizer", revision=None
        )
        super().__init__(service, model_config)
