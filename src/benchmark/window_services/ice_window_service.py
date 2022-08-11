from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class ICEWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        return "TsinghuaKEG/ice"

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input.
        According to https://github.com/THUDM/GLM-130B, the max sequence length is 2048.
        """
        return 2048

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length + 1

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        # TODO: figure out this value. Followed up in https://github.com/THUDM/icetk/issues/1
        return " "

    @property
    def prefix_token(self) -> str:
        """The prefix token"""
        return self.end_of_text_token
