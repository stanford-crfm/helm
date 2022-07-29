from abc import ABC

from .huggingface_window_service import HuggingFaceWindowService
from .tokenizer_service import TokenizerService


class EncoderDecoderWindowService(HuggingFaceWindowService, ABC):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_request_length(self) -> int:
        """
        Return the max request length. We set the max requests length to be `max_sequence_length - 1
        because the encoder appends an additional token ("</s>") when encoding the text.
        """
        return self.max_sequence_length - 1

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).
        """
        return self.get_num_tokens(text) <= self.max_request_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the context window given by `max_request_length`.
        Removes the ending `</s>` after decoding.
        """
        max_length: int = self.max_request_length
        result: str = self.decode(self.encode(text, truncation=True, max_length=max_length).tokens)
        result = result.rstrip("</s>")

        # Validate that the truncated text now fits. Fail fast otherwise.
        num_tokens: int = self.get_num_tokens(result)
        assert num_tokens <= max_length, f"Truncation failed ({num_tokens} > {max_length}). Input text: {text}"
        return result
