from typing import Optional
from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class DefaultWindowService(LocalWindowService):
    def __init__(
        self,
        service: TokenizerService,
        tokenizer_name: str,
        max_sequence_length: int,
        max_request_length: Optional[int] = None,
    ):
        super().__init__(service)
        self._tokenizer_name = tokenizer_name
        self._max_sequence_length = max_sequence_length
        self._max_request_length = max_request_length

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def max_request_length(self) -> int:
        return self._max_request_length or self._max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        # TODO: Support this
        return ""

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def prefix_token(self) -> str:
        # TODO: Support this
        return ""
