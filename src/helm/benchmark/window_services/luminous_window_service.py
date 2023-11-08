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
        """
        From https://docs.aleph-alpha.com/api/complete, "the summed number of tokens of prompt
        and maximum_tokens..may not exceed 2048 tokens." Confirmed it's 2048 for the Luminous
        models currently available.
        """
        return 2048

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
