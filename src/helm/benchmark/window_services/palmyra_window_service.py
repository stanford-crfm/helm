from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class PalmyraWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        """All Palmyra models use the same tokenizer."""
        return "writer/palmyra-tokenizer"

    @property
    def max_sequence_length(self) -> int:
        # Slightly less than the advertised number of 2048,
        # to give us a margin of safety.
        return 2000

    @property
    def max_request_length(self) -> int:
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
        # Slightly less than the advertised number of 8192,
        # to give us a margin of safety.
        return 8000
