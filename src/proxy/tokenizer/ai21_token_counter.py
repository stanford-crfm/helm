from typing import List

from common.request import Request, Sequence
from .openai_token_counter import OpenAITokenCounter
from .token_counter import TokenCounter


class AI21TokenCounter(TokenCounter):
    def __init__(self):
        # We use the OpenAI tokenizer to fit prompts within the context window for two reasons:
        # 1. The tokenizer used for the Jurassic models was not made public. Instead, they have
        #    a tokenizer API, which we want to avoid calling to limit the number of requests we
        #    make to AI21.
        # 2. The Jurassic tokenizer is coarser than the GPT-3 tokenizer, so if the prompt fits
        #    within the GPT-3 context window, it should also fit in the Jurassic context window.
        self.openai_token_counter = OpenAITokenCounter()

    def count_tokens(self, request: Request, completions: List[Sequence]) -> int:
        """
        Counts the number of generated tokens and NOT the number of tokens in the prompt
        (https://studio.ai21.com/docs/calculating-usage).

        The AI21 documentation (https://studio.ai21.com/docs/calculating-usage/) defines
        generated tokens as:
        "the total number of all completion tokens you generate. For example, assume you post
        a complete request for J1-Jumbo with a prompt consisting of 10 tokens and requiring 3
        completions, i.e. numResults = 3, and the model generates completions with 5, 15, and
        20 tokens. In total this request will consume 5+15+20=40 generated tokens."
        """
        return sum(len(sequence.tokens) for sequence in completions)

    def estimate_tokens(self, request: Request) -> int:
        """
        Estimate the number of tokens given a request. We do not need to account for the number
        of tokens in the prompt itself (https://studio.ai21.com/docs/calculating-usage).

        Therefore, estimate using the following formula:

            num_completions * max_tokens
        """
        return request.num_completions * request.max_tokens

    def fits_within_context_window(self, model: str, text: str, expected_completion_token_length: int) -> bool:
        """
        For a given a model and expected token length of the completion, checks if the given text fits within
        the context window using the OpenAITokenCounter.
        """
        return self.openai_token_counter.fits_within_context_window(model, text, expected_completion_token_length)

    def truncate_from_right(self, model: str, text: str) -> str:
        """
        Truncates text from the right using the OpenAITokenCounter.
        """
        return self.openai_token_counter.truncate_from_right(model, text)
