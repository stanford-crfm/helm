from .encoder_decoder_window_service import EncoderDecoderWindowService
from .tokenizer_service import TokenizerService


class UL2WindowService(EncoderDecoderWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length."""
        # From https://arxiv.org/pdf/2205.05131.pdf, "the sequence length is set to 512/512 for inputs and targets".
        # We subtract 1 to account for <extra_id_0> that gets appended to prompts.
        return 512 - 1

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return "</s>"

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "google/ul2"

    @property
    def prefix_token(self) -> str:
        """The prefix token is the same as the end of text token."""
        # echo=True is not supported
        return ""
