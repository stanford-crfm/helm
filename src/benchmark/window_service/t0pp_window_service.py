from .huggingface_window_service import HuggingFaceWindowService
from .tokenizer_service import TokenizerService


class T0ppWindowService(HuggingFaceWindowService):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length."""
        # From https://arxiv.org/pdf/2110.08207.pdf, "we truncate input and target sequences to 1024 and 256 tokens,
        # respectively. Following Raffel et al. (2020), we use packing to combine multiple training examples into
        # a single sequence to reach the maximum sequence length."
        return 512

    @property
    def max_request_length(self) -> int:
        """Return the max request length."""
        return self.max_sequence_length

    @property
    def end_of_text_token(self) -> str:
        """The end of text token."""
        # TODO: @YianZhang - double check this
        return ""

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "bigscience/T0pp"

    @property
    def prefix_token(self) -> str:
        """The prefix token is the same as the end of text token."""
        return self.end_of_text_token
