from helm.proxy.clients.huggingface_client import HuggingFaceModelConfig
from helm.benchmark.window_services.huggingface_window_service import HuggingFaceWindowService
from helm.benchmark.window_services.tokenizer_service import TokenizerService


class LlamaWindowService(HuggingFaceWindowService):
    def __init__(self, service: TokenizerService):
        # Loading this Hugging FAce tokenizer takes 2 minutes, mostly due to convert_slow_tokenizer.
        # TODO: Make this faster.
        model_config = HuggingFaceModelConfig(namespace="tatsu-lab", model_name="alpaca-7b-wdiff", revision=None)
        super().__init__(service, model_config)
        # tatsu-lab/alpaca-7b-wdiff incorrectly gives a sequence length of 512.
        # The LLaMA max sequence length is 2048.
        # See: https://github.com/facebookresearch/llama/issues/16
        self._max_sequence_length = 512
