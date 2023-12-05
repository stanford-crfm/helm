from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class PaLM2WindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        """The tokenizer is most likely not correct but there is no official tokenizer.
        See comment in model_deployments.yaml for more info."""
        return "google/t5-11b"

    @property
    def max_sequence_length(self) -> int:
        return 8192

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        return super().end_of_text_token

    @property
    def prefix_token(self) -> str:
        return super().prefix_token


class PaLM232KWindowService(PaLM2WindowService):
    @property
    def max_sequence_length(self) -> int:
        return 32000

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self.max_request_length


class CodeBisonWindowService(PaLM2WindowService):
    @property
    def max_sequence_length(self) -> int:
        return 6144
