import os
import pytest
import tempfile

from helm.common.cache import SqliteCacheConfig
from helm.common.request import Request, RequestResult
from .huggingface_client import HuggingFaceClient


class TestHuggingFaceClient:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.client = HuggingFaceClient(cache_config=SqliteCacheConfig(self.cache_path))

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_gpt2(self):
        prompt: str = "I am a computer scientist."
        result: RequestResult = self.client.make_request(
            Request(
                model="openai/gpt2",
                model_deployment="huggingface/gpt2",
                prompt=prompt,
                num_completions=3,
                top_k_per_token=5,
                max_tokens=0,
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
                model="eleutherai/gpt-j-6b",
                model_deployment="huggingface/gpt-j-6b",
                prompt="I am a computer scientist.",
                num_completions=3,
                top_k_per_token=5,
                max_tokens=0,
            )
        )
        assert len(result.completions) == 3

    def test_logprob(self):
        prompt: str = "I am a computer scientist."
        result: RequestResult = self.client.make_request(
            Request(
                model="openai/gpt2",
                model_deployment="huggingface/gpt2",
                prompt=prompt,
                num_completions=1,
                max_tokens=0,
                echo_prompt=True,
            )
        )
        assert result.completions[0].text.startswith(
            prompt
        ), "echo_prompt was set to true. Expected the prompt at the beginning of each completion"
        total_logprob: float = 0
        assert len(result.completions[0].tokens) == 6, "Expected 6 tokens in the completion"
        for token in result.completions[0].tokens[1:]:
            assert token.logprob != 0
            total_logprob += token.logprob
        assert result.completions[0].logprob == pytest.approx(total_logprob)
