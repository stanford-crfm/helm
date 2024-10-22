from abc import ABC

from helm.common.hierarchical_logger import hlog
from helm.benchmark.window_services.local_window_service import LocalWindowService


class EncoderDecoderWindowService(LocalWindowService, ABC):
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
        if expected_completion_token_length > self.max_output_length:
            hlog(
                f"WARNING: The expected completion token length ({expected_completion_token_length}) "
                f"exceeds the max output length ({self.max_output_length})."
            )
        return self.get_num_tokens(text) <= self.max_request_length

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to left to fit within the maximum context length given
        by `max_request_length`. Does not take into expected completion length because the
        the encoder-decoder models have separate max context lengths for the input prompts
        and completions.
        """
        result: str = self.decode(self.encode(text, truncation=True, max_length=self.max_request_length).tokens)

        # HACK: For the vast majority of cases, the above logic works, but there are a few where the
        # token count exceeds `max_length` by 1.
        while not self.fits_within_context_window(result):
            result = result[:-1]

        return result
