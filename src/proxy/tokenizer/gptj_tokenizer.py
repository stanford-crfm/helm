from transformers import GPT2TokenizerFast

from .gpt2_tokenizer import GPT2Tokenizer


class GPTJTokenizer(GPT2Tokenizer):

    # The max length of the model input. The max sequence length for GPT-J is 2048.
    MAX_SEQUENCE_LENGTH: int = 2048

    # TODO: Figure out how to do language modeling with GPT-J
    # The length of the returned text (prompt + generation) cannot be longer than
    # 2048 tokens. `max_tokens` cannot be zero. The logprob of the first token is
    # not returned.
    MAX_REQUEST_LENGTH: int = 2048

    def __init__(self, tokenizer: GPT2TokenizerFast, cache_path: str):
        # Use the GPT-2 tokenizer.
        # From https://huggingface.co/EleutherAI/gpt-j-6B, "the model [GPT-J] is trained with a
        # tokenization vocabulary of 50257, using the same set of BPEs as GPT-2/GPT-3".
        super().__init__(tokenizer, cache_path)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length for GPT-J."""
        return GPTJTokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length for GPT-J."""
        return GPTJTokenizer.MAX_REQUEST_LENGTH
