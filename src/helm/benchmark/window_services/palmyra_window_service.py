from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class PalmyraWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        """All Palmyra models use the same tokenizer."""
        # Palmyra models do not support echo
        # So they have a different TokenizerConfig called "writer/gpt2"
        # when in reality they use the same tokenizer as "huggingface/gpt2"
        return "writer/gpt2"

    @property
    def max_sequence_length(self) -> int:
        return 2048

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        """
        The end of text token.
        TODO: Setting to empty string for now as echo is not supported.
        """
        return ""

    @property
    def prefix_token(self) -> str:
        """
        The prefix token.
        """
        return self.end_of_text_token


class LongerPalmyraWindowService(PalmyraWindowService):
    @property
    def max_sequence_length(self) -> int:
        return 8192


class Palmyra6KWindowService(PalmyraWindowService):
    @property
    def max_sequence_length(self) -> int:
        return 6000

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self.max_sequence_length + 1024


class Palmyra32KWindowService(PalmyraWindowService):
    @property
    def max_sequence_length(self) -> int:
        return 28000

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self.max_sequence_length + 2048
