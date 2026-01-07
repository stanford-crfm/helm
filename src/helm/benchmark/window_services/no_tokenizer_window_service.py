from typing import List, Optional
from helm.benchmark.window_services.local_window_service import LocalWindowService
from helm.benchmark.window_services.window_service import EncodeResult
from helm.common.tokenization_request import TokenizationToken


_INT_MAX: int = 2**31 - 1


class NoTokenizerWindowService(LocalWindowService):
    _MAX_LENGTH = 1_000_000_000

    @property
    def tokenizer_name(self) -> str:
        return "tokenizer/no-op"

    @property
    def max_sequence_length(self) -> int:
        return _INT_MAX

    @property
    def max_request_length(self) -> int:
        return _INT_MAX

    @property
    def end_of_text_token(self) -> str:
        return ""

    @property
    def prefix_token(self) -> str:
        return ""

    def encode(self, text: str, truncation: bool = False, max_length: Optional[int] = None) -> EncodeResult:
        raise NotImplementedError("NoOpWindowService.encode is not supported")

    def decode(self, tokens: List[TokenizationToken], normalized_text: Optional[str] = None) -> str:
        raise NotImplementedError("NoOpWindowService.decode is not supported")

    def tokenize(self, text: str) -> List[str]:
        return [text]

    def get_num_tokens(self, text: str) -> int:
        return 1

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        return True

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        return text
