from .ai21_window_service import AI21WindowService


class WiderAI21WindowService(AI21WindowService):
    @property
    def max_sequence_length(self) -> int:
        """
        Return the max sequence length of the larger AI21 Jurassic-2 models.

        The AI21 server automatically prepends a token to every prompt,
        so the actual max sequence length is 8192 - 1 = 8191.
        """
        return 8191
