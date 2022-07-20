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

    @property
    def max_request_length(self) -> int:
        """
        Return the max request length of the OpenAI models.
        From https://help.openai.com/en/articles/5072518-controlling-the-length-of-completions,
        "these requests can use up to 2049 tokens, shared between prompt and completion."
        """
        return 2049
