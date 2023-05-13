from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class PalmyraWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        """All Palmyra models use the same tokenizer."""
        return "Writer/palmyra-base"

    @property
    def max_sequence_length(self) -> int:
        return 2048

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        """
        Return the max prompt length + max token length.
        Writer is one of the rare models that has a limit on this.
        The official limit seems to be 2048, we limit to 2000 just to be safe.
        """
        # TODO #1558: Replace by the exact limit here.
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


class SilkRoadWindowService(PalmyraWindowService):
    @property
    def max_sequence_length(self) -> int:
        return 80000

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        # TODO #1558: Figure out the exact number, should be 4096
        return 4000
