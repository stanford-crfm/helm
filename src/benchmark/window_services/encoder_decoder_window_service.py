from abc import ABC

from .local_window_service import LocalWindowService
from .tokenizer_service import TokenizerService


class EncoderDecoderWindowService(LocalWindowService, ABC):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_request_length(self) -> int:
        """
        Return the max request length. We set the max requests length to be `max_sequence_length`.
        """
        return self.max_sequence_length

    @property
    def max_output_length(self) -> int:
        """
        Return the max output length. Since the encoder-decoder models have separate maximum context lengths
        for the input prompts and the completions, we need to keep track of the two values separately.
        By default, `max_output_length` equals `max_sequence_length`.
        """
        return self.max_sequence_length

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the context window given by `max_request_length`.
        Since the encoder-decoder models have separate maximum context lengths for the input prompts
        vs. the completions, we check the two values separately.
        """
        assert expected_completion_token_length <= self.max_output_length, (
            f"The expected completion token length ({expected_completion_token_length}) exceeds the "
            f"max output length ({self.max_output_length})."
        )
        return self.get_num_tokens(text) <= self.max_request_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to left to fit within the maximum context length given
        by `max_request_length`. Removes the "</s>" that was added when encoding after decoding.
        """
        result: str = self.decode(self.encode(text, truncation=True, max_length=self.max_request_length).tokens)
        if result.endswith("</s>"):
            # Remove the added "</s>"
            result = result[:-4]

        # HACK: For the vast majority of cases, the above logic works, but there are a few where the
        # token count exceeds `max_length` by 1.
        while not self.fits_within_context_window(result):
            result = result[:-1]

        return result
