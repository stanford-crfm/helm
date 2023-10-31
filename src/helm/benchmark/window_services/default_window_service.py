from typing import Optional
from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


_INT_MAX: int = 2147483647


class DefaultWindowService(LocalWindowService):
    def __init__(
        self,
        service: TokenizerService,
        tokenizer_name: str,
        max_sequence_length: int,
        max_request_length: Optional[int] = None,
        end_of_text_token: Optional[str] = None,
        prefix_token: Optional[str] = None,
        max_sequence_and_generated_tokens_length: Optional[int] = None,
    ):
        super().__init__(service)
        self._tokenizer_name = tokenizer_name
        self._max_sequence_length = max_sequence_length
        self._max_request_length = max_request_length or max_sequence_length
        self._end_of_text_token = end_of_text_token or ""
        self._prefix_token = prefix_token or ""
        self._max_sequence_and_generated_tokens_length = max_sequence_and_generated_tokens_length or _INT_MAX

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def max_request_length(self) -> int:
        return self._max_request_length

    @property
    def end_of_text_token(self) -> str:
        return self._end_of_text_token

    @property
    def prefix_token(self) -> str:
        return self._prefix_token

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self._max_sequence_and_generated_tokens_length
