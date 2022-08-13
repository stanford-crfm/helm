from proxy.clients.yalm_tokenizer.yalm_tokenizer import YaLMTokenizer
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
