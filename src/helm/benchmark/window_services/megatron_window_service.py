from .gpt2_window_service import GPT2WindowService


# NOTE: The only difference between this and GPT2WindowService is that
# the request length is constrained to the sequence length.
class MegatronWindowService(GPT2WindowService):
    @property
    def max_request_length(self) -> int:
        """Return the max request length of GPT-2."""
        return self.max_sequence_length
