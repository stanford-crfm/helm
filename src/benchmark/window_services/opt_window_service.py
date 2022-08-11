from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class OPTWindowService(LocalWindowService):
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
        """Return the max request length."""
        return self.max_sequence_length + 1

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return "</s>"

    @property
    def prefix_token(self) -> str:
        """The prefix token is the same as the end of text token."""
        return self.end_of_text_token

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "facebook/opt-66b"
