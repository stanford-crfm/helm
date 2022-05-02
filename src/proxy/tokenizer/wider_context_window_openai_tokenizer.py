from transformers import GPT2TokenizerFast

from .gpt2_tokenizer import GPT2Tokenizer


class WiderContextWindowOpenAITokenizer(GPT2Tokenizer):

    # From https://beta.openai.com/docs/engines
    MAX_SEQUENCE_LENGTH: int = 4000

    def __init__(self, tokenizer: GPT2TokenizerFast):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        super().__init__(tokenizer)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the OpenAI models."""
        return WiderContextWindowOpenAITokenizer.MAX_SEQUENCE_LENGTH

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the OpenAI max sequence length of 4000 taking to account
        the expected completion length (defaults to 0).
        """
        return (
            self.tokenize_and_count(text) + expected_completion_token_length
            <= WiderContextWindowOpenAITokenizer.MAX_SEQUENCE_LENGTH
        )

    def truncate_from_right(self, text: str, expected_completion_token_length: int = 0) -> str:
        """
        Truncates text from the right to fit within the OpenAI max sequence length of 4000
        minus the expected completion length (defaults to 0).
        """
        return self._tokenizer.decode(
            self._tokenizer.encode(
                text,
                truncation=True,
                max_length=WiderContextWindowOpenAITokenizer.MAX_SEQUENCE_LENGTH - expected_completion_token_length,
            )
        )
