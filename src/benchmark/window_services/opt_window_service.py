from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class OPTWindowService(GPT2WindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input. The max sequence length for the OPT models is 2048.
        Source: https://arxiv.org/pdf/2205.01068.pdf
        """
        return 2048

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length
