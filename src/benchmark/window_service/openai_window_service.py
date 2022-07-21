from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class OpenAIWindowService(GPT2WindowService):
    def __init__(self, service: TokenizerService):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the OpenAI models (max length of model input)."""
        return 2048
