from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


# TODO: Remove Once we have configurable model names since this hardcodes the tokenizer name
class HTTPModelWindowServce(LocalWindowService):
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
        return "<|endoftext|>"

    @property
    def tokenizer_name(self) -> str:
        return "neurips/local"

    @property
    def prefix_token(self) -> str:
        return self.end_of_text_token
