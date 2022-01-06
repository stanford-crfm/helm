from typing import List

from common.request import Request, Sequence
from proxy.tokenizer.token_counter import TokenCounter
from transformers import GPT2TokenizerFast


class OpenAITokenCounter(TokenCounter):
    def __init__(self):
        # OpenAI used the same tokenizer for GPT-2 and GPT-3.
        # Weights are cached at ~/.cache/huggingface/transformers.
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the total number of tokens using the suggestion here:
        https://community.openai.com/t/how-do-i-calculate-the-pricing-for-generation-of-text/11662/5

        TODO: OpenAI will support counting the number of tokens for us. Adapt this method accordingly.
              https://github.com/stanford-crfm/benchmarking/issues/59
        """
        n_tokens = len(self.tokenizer.encode(request.prompt))
        for sequence in completions:
            n_tokens += len(sequence.tokens)
        return n_tokens
