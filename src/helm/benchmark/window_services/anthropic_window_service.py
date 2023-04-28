from .gpt2_window_service import GPT2WindowService
from .tokenizer_service import TokenizerService


class LegacyAnthropicWindowService(GPT2WindowService):
    """
    This is the window service for the legacy Anthropic client.
    Please consider using the new Anthropic client with the AnthropicWindowService.
    Anthropic has their own tokenizer based on the GPT-2 tokenizer.
    This tokenizer is not publicly available, so approximate with the GPT-2 tokenizer.
    """

    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the Anthropic model."""
        return 8192

    @property
    def max_request_length(self) -> int:
        """
        Return the max request length of the Anthropic model.
        Anthropic does not include the start of sequence token.
        """
        return self.max_sequence_length


class AnthropicWindowService(GPT2WindowService):
    """
    Anthropic has their own tokenizer.
    """

    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        Return the max sequence length of the Anthropic model.
        While the limits seems to be 8192, we limit to 8000
        according to Anthropic's recommendations.
        See: https://console.anthropic.com/docs/prompt-design
        """
        return 8000

    @property
    def max_request_length(self) -> int:
        """
        Return the max request length of the Anthropic model.
        Anthropic does not include the start of sequence token.
        """
        return self.max_sequence_length

    @property
    def tokenizer_name(self) -> str:
        return "anthropic/claude"
