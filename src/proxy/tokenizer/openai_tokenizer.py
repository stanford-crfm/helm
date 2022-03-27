from .gpt2_tokenizer import GPT2Tokenizer


class OpenAITokenizer(GPT2Tokenizer):

    # From https://help.openai.com/en/articles/5072518-controlling-the-length-of-completions,
    # "these requests can use up to 2049 tokens, shared between prompt and completion."
    MAX_REQUEST_LENGTH: int = 2049

    # The max length of the model input. The max sequence length for OpenAI is 2048,
    # which is different from the max request length of 2049.
    MAX_SEQUENCE_LENGTH: int = 2048

    def __init__(self):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        super().__init__()

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the OpenAI models."""
        return OpenAITokenizer.MAX_SEQUENCE_LENGTH

    def fits_within_context_window(self, text: str, expected_completion_token_length: int = 0) -> bool:
        """
        Checks if the given text fits within the OpenAI max request length of 2049 taking to account
        the expected completion length (defaults to 0).
        """
        return self.tokenize_and_count(text) + expected_completion_token_length <= OpenAITokenizer.MAX_REQUEST_LENGTH

    def truncate_from_right(self, text: str) -> str:
        """
        Truncates text from the right to fit within the OpenAI max request length of 2049.
        """
        return self._tokenizer.decode(
            self._tokenizer.encode(text, truncation=True, max_length=OpenAITokenizer.MAX_REQUEST_LENGTH)
        )
