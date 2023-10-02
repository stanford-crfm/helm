from abc import ABC

from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class IDEFICSWindowService(LocalWindowService, ABC):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        return 2048

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        return "</s>"

    @property
    def prefix_token(self) -> str:
        return self.end_of_text_token


class IDEFICS9bWindowService(IDEFICSWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "HuggingFaceM4/idefics-9b"


class IDEFICS9bInstructWindowService(IDEFICSWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "HuggingFaceM4/idefics-9b-instruct"


class IDEFICS80bWindowService(IDEFICSWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "HuggingFaceM4/idefics-80b"


class IDEFICS80bInstructWindowService(IDEFICSWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "HuggingFaceM4/idefics-80b-instruct"
