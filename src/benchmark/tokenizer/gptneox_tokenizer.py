from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class GPTNeoXTokenizer(GPT2Tokenizer):

    # The max length of the model input. The max sequence length for GPT-NeoX is 2048.
    MAX_SEQUENCE_LENGTH: int = 2048

    # The max request length of GPT-NeoX is MAX_SEQUENCE_LENGTH + 1.
    MAX_REQUEST_LENGTH: int = 2049

    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length."""
        return GPTNeoXTokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length."""
        return GPTNeoXTokenizer.MAX_REQUEST_LENGTH

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "huggingface/gpt-neox-20b"
