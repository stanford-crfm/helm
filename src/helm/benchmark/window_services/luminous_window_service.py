from abc import abstractmethod

from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class LuminousWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    @abstractmethod
    def tokenizer_name(self) -> str:
        """Each Luminous model has its own tokenizer."""
        pass

    @property
    def max_sequence_length(self) -> int:
        return self.max_request_length

    @property
    def max_request_length(self) -> int:
        """
        The max request length according to https://docs.aleph-alpha.com/api/complete.
        TODO: double check if this is only for the completion (not including the prompt).
        """
        return 2048

    @property
    def end_of_text_token(self) -> str:
        """
        The end of text token.
        TODO: echo doesn't seem to be supported.
        """
        return ""

    @property
    def prefix_token(self) -> str:
        """
        The prefix token.
        TODO: echo doesn't seem to be supported.
        """
        return self.end_of_text_token


class LuminousBaseWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-base"


class LuminousExtendedWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-extended"


class LuminousSupremeWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-supreme"


class LuminousWorldWindowService(LuminousWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "AlephAlpha/luminous-world"
