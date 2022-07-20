from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class GPTJWindowService(GPT2WindowService):
    """
    The same tokenizer as GPT-2, but with an additional 143 tokens
    (source: https://huggingface.co/docs/transformers/model_doc/gptj).
    """

    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length."""
        return 2048

    @property
    def max_request_length(self) -> int:
        """Return the max request length."""
        return 2049

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "huggingface/gpt-j-6b"
