from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class GPTJWindowService(LocalWindowService):
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
        return self.max_sequence_length + 1

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "EleutherAI/gpt-j-6B"

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return "<|endoftext|>"

    @property
    def prefix_token(self) -> str:
        """The prefix token for models is the same as the end of text token."""
        return self.end_of_text_token
