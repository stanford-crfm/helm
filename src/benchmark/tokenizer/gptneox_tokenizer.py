from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class GPTNeoXTokenizer(GPT2Tokenizer):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length."""
        return 2048

    @property
    def max_request_length(self) -> int:
        """
        Return the max request length. The max request length of GPT-NeoX is 1 greater
        than the max sequence length.
        """
        return 2049

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "huggingface/gpt-neox-20b"
