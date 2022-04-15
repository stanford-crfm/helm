from transformers import GPT2TokenizerFast

from .gpt2_tokenizer import GPT2Tokenizer


class MTNLGTokenizer(GPT2Tokenizer):

    # The max length of the model input. The max sequence length for the MT-NLG models is 2048.
    # Source: https://github.com/microsoft/turing-academic-TNLG
    MAX_SEQUENCE_LENGTH: int = 2048

    def __init__(self, tokenizer: GPT2TokenizerFast):
        # Use the GPT-2 tokenizer.
        super().__init__(tokenizer)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length for the MT-NLG models."""
        return MTNLGTokenizer.MAX_SEQUENCE_LENGTH
