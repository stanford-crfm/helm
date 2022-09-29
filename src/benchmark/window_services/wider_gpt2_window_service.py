from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class WiderGPT2WindowService(GPT2WindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of this tokenizer."""
        return 2048

    @property
    def max_request_length(self) -> int:
        """Return the max request length of GPT-FewShot"""
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return "<|endoftext|>"

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "huggingface/gpt2"

    @property
    def prefix_token(self) -> str:
        """The prefix token for models that uses the GPT-2 tokenizer is the end of text token."""
        return self.end_of_text_token
