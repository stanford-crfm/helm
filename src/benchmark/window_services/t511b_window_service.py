from .encoder_decoder_window_service import EncoderDecoderWindowService
from .tokenizer_service import TokenizerService


class T511bWindowService(EncoderDecoderWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length."""
        # From https://arxiv.org/pdf/1910.10683.pdf, "we use a maximum sequence length of 512".
        # We subtract 1 to account for <extra_id_0> that gets appended to prompts.
        return 512 - 1

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        return "</s>"

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "google/t5-11b"

    @property
    def prefix_token(self) -> str:
        """The prefix token is the same as the end of text token."""
        # echo=True is not supported
        return ""
