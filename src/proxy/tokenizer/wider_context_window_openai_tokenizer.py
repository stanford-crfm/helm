from transformers import GPT2TokenizerFast

from .gpt2_tokenizer import GPT2Tokenizer


class WiderContextWindowOpenAITokenizer(GPT2Tokenizer):

    # From https://beta.openai.com/docs/engines
    MAX_SEQUENCE_LENGTH: int = 4000

    # Note that the API doesn't actually raise an error when the prompt is longer than 4001 tokens.
    # Set this value to 4001 for consistency with MAX_SEQUENCE_LENGTH.
    MAX_REQUEST_LENGTH: int = 4001

    def __init__(self, tokenizer: GPT2TokenizerFast):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        super().__init__(tokenizer)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the larger second-generation OpenAI models."""
        return WiderContextWindowOpenAITokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length of the larger second-generation OpenAI models."""
        return WiderContextWindowOpenAITokenizer.MAX_REQUEST_LENGTH
