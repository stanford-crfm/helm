from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class AnthropicTokenizer(GPT2Tokenizer):
    """
    Anthropic has their own tokenizer based on the GPT-2 tokenizer.
    This tokenizer is not publicly available, so approximate with the GPT-2 tokenizer.
    """

    # The Anthropic model has max sequence length of 8192.
    MAX_SEQUENCE_LENGTH: int = 8192

    # TODO: Once the model supports returning logprobs,
    # find the correct value for `MAX_REQUEST_LENGTH`
    MAX_REQUEST_LENGTH: int = 8192

    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the Anthropic model."""
        return AnthropicTokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length of the Anthropic model."""
        return AnthropicTokenizer.MAX_REQUEST_LENGTH
