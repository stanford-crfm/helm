from transformers import GPT2TokenizerFast

from .gpt2_tokenizer import GPT2Tokenizer


class AnthropicTokenizer(GPT2Tokenizer):

    # The Anthropic model has max sequence length of 8192.
    MAX_SEQUENCE_LENGTH: int = 8192

    def __init__(self, tokenizer: GPT2TokenizerFast):
        # Anthropic has their own tokenizer based on the GPT-2 tokenizer.
        # This tokenizer is not publicly available, so approximate with the GPT-2 tokenizer.
        super().__init__(tokenizer)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the Anthropic model."""
        return AnthropicTokenizer.MAX_SEQUENCE_LENGTH
