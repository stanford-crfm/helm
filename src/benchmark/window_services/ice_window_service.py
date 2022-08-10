from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class ICEWindowService(LocalWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def tokenizer_name(self) -> str:
        return "TsinghuaKEG/ice"

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input.
        According to https://github.com/THUDM/GLM-130B, the max sequence length is 2048.
        """
        return 2048

    @property
    def max_request_length(self) -> int:
        return self.max_sequence_length + 1

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        # TODO: figure out this value. Followed up in https://github.com/THUDM/icetk/issues/1
        return " "

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
        result: str = self.decode(self.encode(text).tokens[:max_length])

        # Validate that the truncated text now fits. Fail fast otherwise.
        num_tokens: int = self.get_num_tokens(result)
        assert num_tokens <= max_length, f"Truncation failed ({num_tokens} > {max_length}). Input text: {text}"
        return result
