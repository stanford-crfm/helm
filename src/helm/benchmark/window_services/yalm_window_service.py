from helm.proxy.clients.yalm_tokenizer.yalm_tokenizer import YaLMTokenizer
from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class YaLMWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        return "Yandex/yalm"

    @property
    def max_sequence_length(self) -> int:
        return YaLMTokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length + 1

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return YaLMTokenizer.EOS_TOKEN

    @property
    def prefix_token(self) -> str:
        """The prefix token"""
        return self.end_of_text_token

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`
        minus the expected completion length (defaults to 0).
        """
        max_length: int = self.max_request_length - expected_completion_token_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but it sometimes doesn't work
        # for certain cases
        # (e.g., Tamil script from copyright:datatag=n_books_1000-extractions_per_book_1-prefix_length_125).
        # Truncate by removing character by character until the prompt fits within the context window.
        while not self.fits_within_context_window(result, expected_completion_token_length):
            result = result[:-1]

        return result
