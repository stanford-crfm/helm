from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class PaLM2WindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        """The tokenizer is most likely not correct but there is no official tokenizer.
        See comment in model_deployments.yaml for more info."""
        # TODO #2083: Update this when the tokenizer is known.
        return "google/mt5-base"

    @property
    def max_sequence_length(self) -> int:
        return 6000  # Officially 8192

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self.max_sequence_length + 1000  # Officially 1024

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        # TODO #2083: Update this when the tokenizer is known.
        # This is purely a guess based on T511bWindowService.
        return "</s>"

    @property
    def prefix_token(self) -> str:
        # TODO #2083: Update this when the tokenizer is known.
        # This is purely a guess based on T511bWindowService.
        return ""


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
        return 6000  # Officially 6144

    @property
    def max_sequence_and_generated_tokens_length(self) -> int:
        return self.max_request_length + 1000  # Officially 1024
