import os
import pytest
import tempfile

from helm.common.cache import SqliteCacheConfig
from helm.common.request import Request, RequestResult
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from .huggingface_client import HuggingFaceClient


class TestHuggingFaceClient:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.client = HuggingFaceClient(SqliteCacheConfig(self.cache_path))

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_tokenize(self):
        request = TokenizationRequest(text="I am a computer scientist.")
        result: TokenizationRequestResult = self.client.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        result: TokenizationRequestResult = self.client.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == ["I", " am", " a", " computer", " scientist", "."]

    def test_encode(self):
        request = TokenizationRequest(text="I am a computer scientist.", encode=True, truncation=True, max_length=1)
        result: TokenizationRequestResult = self.client.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        result: TokenizationRequestResult = self.client.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == [40]

        request = TokenizationRequest(text="I am a computer scientist.", encode=True, truncation=True, max_length=1024)
        result = self.client.tokenize(request)
        assert not result.cached, "First time making this particular request. Result should not be cached"
        assert result.raw_tokens == [40, 716, 257, 3644, 11444, 13]

    def test_decode(self):
        request = DecodeRequest(tokens=[40, 716, 257, 3644, 11444, 13])
        result: DecodeRequestResult = self.client.decode(request)
        assert not result.cached, "First time making the decode request. Result should not be cached"
        result: DecodeRequestResult = self.client.decode(request)
        assert result.cached, "Result should be cached"
        assert result.text == "I am a computer scientist."

    def test_gpt2(self):
        prompt: str = "I am a computer scientist."
        result: RequestResult = self.client.make_request(
            Request(
                model="huggingface/gpt2",
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
                model="huggingface/gpt-j-6b",
                prompt="I am a computer scientist.",
                num_completions=3,
                top_k_per_token=5,
                max_tokens=0,
            )
        )
        assert len(result.completions) == 3
