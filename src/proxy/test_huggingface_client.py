import pytest
import os
import tempfile

from .huggingface_client import HuggingFaceClient
from common.request import Request, RequestResult


class TestHuggingFaceClient:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.client = HuggingFaceClient(cache_path=self.cache_path)

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_gpt2(self):
        prompt: str = "I am a computer scientist."
        result: RequestResult = self.client.make_request(
            Request(
                model="huggingface/gpt2",
                prompt=prompt,
                num_completions=3,
                max_tokens=20,
                top_k_per_token=5,
                stop_sequences=["\n"],
                echo_prompt=True,
            )
        )
        assert len(result.completions) == 3
        assert result.completions[0].text.startswith(
            prompt
        ), "echo_prompt was set to true. Expected the prompt at the beginning of each completion"

    @pytest.mark.skip(reason="GPT-J 6B is 22 GB and extremely slow without a GPU.")
    def test_gptj_6b(self):
        result: RequestResult = self.client.make_request(
            Request(
                model="huggingface/gptj_6b",
                prompt="I am a computer scientist.",
                num_completions=3,
                max_tokens=20,
                top_k_per_token=5,
            )
        )
        assert len(result.completions) == 3
