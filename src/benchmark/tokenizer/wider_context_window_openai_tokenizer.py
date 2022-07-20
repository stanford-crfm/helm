from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class WiderContextWindowOpenAITokenizer(GPT2Tokenizer):
    def __init__(self, service: TokenizerService):
        # OpenAI uses the same tokenizer for GPT-2 and GPT-3.
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """
        Return the max sequence length of the larger second-generation OpenAI models.
        Source: https://beta.openai.com/docs/engines
        """
        return 4000

    @property
    def max_request_length(self) -> int:
        """
        Return the max request length of the larger second-generation OpenAI models.
        """
        # Note that the API doesn't actually raise an error when the prompt is longer than 4001 tokens.
        # Set this value to 4001 for consistency with MAX_SEQUENCE_LENGTH
        return 4001
