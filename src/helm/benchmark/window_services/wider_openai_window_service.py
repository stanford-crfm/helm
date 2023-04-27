from .gpt2_window_service import GPT2WindowService


class WiderOpenAIWindowService(GPT2WindowService):
    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the larger second-generation OpenAI models.

        Source: https://platform.openai.com/docs/models"""
        return 4000


class OpenAIChatWindowService(WiderOpenAIWindowService):
    @property
    def tokenizer_name(self) -> str:
        return "openai/cl100k_base"


class GPT3Point5TurboWindowService(OpenAIChatWindowService):
    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length for GPT-3.5 Turbo.

        Source: https://platform.openai.com/docs/models"""
        return 4000


class GPT4WindowService(OpenAIChatWindowService):
    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length for GPT-4.

        Source: https://platform.openai.com/docs/models"""
        return 8192


class GPT432KWindowService(OpenAIChatWindowService):
    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length for GPT-4.

        Source: https://platform.openai.com/docs/models"""
        return 32768
