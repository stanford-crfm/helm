from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class YaLMWindowService(GPT2WindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input.
        """
        return 2048

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length
