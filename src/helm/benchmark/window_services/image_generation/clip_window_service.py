from abc import ABC

from helm.benchmark.window_services.local_window_service import LocalWindowService
from helm.benchmark.window_services.tokenizer_service import TokenizerService


class CLIPWindowService(LocalWindowService, ABC):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        The max length is 77, but we also need to account for <|startoftext|> and <|endoftext|>."
        """
        return 77 - 2

    @property
    def max_request_length(self) -> int:
        """Return the max request length (same as `max_sequence_length`)."""
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        return ""

    @property
    def prefix_token(self) -> str:
        return self.end_of_text_token

    @property
    def tokenizer_name(self) -> str:
        return "openai/clip-vit-large-patch14"

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        result: str = self.decode(self.encode(text, truncation=True, max_length=self.max_request_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but there are a few where the
        # token count exceeds `max_length` by 1.
        while not self.fits_within_context_window(result):
            result = result[:-1]

        return result
