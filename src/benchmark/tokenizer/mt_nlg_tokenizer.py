from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class MTNLGTokenizer(GPT2Tokenizer):

    # The max length of the model input. The max sequence length for the MT-NLG models is 2048.
    # Source: https://github.com/microsoft/turing-academic-TNLG
    MAX_SEQUENCE_LENGTH: int = 2048

    # The max request length of the MT-NLG model is MAX_SEQUENCE_LENGTH + 1.
    MAX_REQUEST_LENGTH: int = 2049

    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length for the MT-NLG models."""
        return MTNLGTokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length for the MT-NLG models."""
        return MTNLGTokenizer.MAX_REQUEST_LENGTH
