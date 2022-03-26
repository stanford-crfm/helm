import pytest
import shutil
import tempfile

from proxy.huggingface_client import HuggingFaceClient
from common.request import Request, RequestResult


class TestHuggingFaceClient:
    def setup_method(self, method):
        self.cache_dir = tempfile.mkdtemp()
        self.client = HuggingFaceClient(cache_path=self.cache_dir)

    def teardown_method(self, method):
        shutil.rmtree(self.cache_dir)

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
        assert len(result.completions[0].text)

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
