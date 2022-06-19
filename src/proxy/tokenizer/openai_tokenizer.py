from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class OpenAITokenizer(GPT2Tokenizer):

    # From https://help.openai.com/en/articles/5072518-controlling-the-length-of-completions,
    # "these requests can use up to 2049 tokens, shared between prompt and completion."
    MAX_REQUEST_LENGTH: int = 2049

    # The max length of the model input. The max sequence length for OpenAI is 2048,
    # which is different from the max request length of 2049.
    MAX_SEQUENCE_LENGTH: int = 2048

    def __init__(self, service: TokenizerService):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length of the OpenAI models."""
        return OpenAITokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length of the OpenAI models."""
        return OpenAITokenizer.MAX_REQUEST_LENGTH
