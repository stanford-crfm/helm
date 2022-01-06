from typing import List

from common.request import Request, Sequence
from .openai_token_counter import OpenAITokenCounter


class TestOpenAITokenCounter:
    def setup_method(self, method):
        self.token_counter = OpenAITokenCounter()

    def test_count_tokens(self):
        request = Request(
            prompt="The Center for Research on Foundation Models (CRFM) is "
            "an interdisciplinary initiative born out of the Stanford "
            "Institute for Human-Centered Artificial Intelligence (HAI) "
            "that aims to make fundamental advances in the study, development, "
            "and deployment of foundation models."
        )
        completions: List[Sequence] = []

        # Verified against https://beta.openai.com/tokenizer
        assert self.token_counter.count_tokens(request, completions) == 51
