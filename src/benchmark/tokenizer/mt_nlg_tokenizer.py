from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class MTNLGTokenizer(GPT2Tokenizer):
    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        The max length of the model input. The max sequence length for the MT-NLG models is 2048.
        Source: https://github.com/microsoft/turing-academic-TNLG
        """
        return 2048

    @property
    def max_request_length(self) -> int:
        """Return the max request length for the MT-NLG models."""
        # TODO: double check this
        return 2049
