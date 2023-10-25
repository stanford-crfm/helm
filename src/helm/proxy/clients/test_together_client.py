import os
import pytest
import tempfile

from helm.common.cache import SqliteCacheConfig
from helm.common.request import Request
from helm.proxy.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer

from .together_client import TogetherClient, TogetherClientError


class TestTogetherClient:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.client = TogetherClient(
            tokenizer=HuggingFaceTokenizer(SqliteCacheConfig(self.cache_path)),
            cache_config=SqliteCacheConfig(self.cache_path),
        )

    def teardown_method(self, method):
        os.remove(self.cache_path)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Request(
                    model="together/redpajama-incite-base-3b-v1",
                ),
                {
                    "best_of": 1,
                    "echo": False,
                    "logprobs": 1,
                    "max_tokens": 100,
                    "model": "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                    "n": 1,
                    "prompt": "",
                    "request_type": "language-model-inference",
                    "stop": None,
                    "temperature": 1.0,
                    "top_p": 1,
                },
            ),
            (
                Request(
                    model="meta/llama-7b",
                    prompt="I am a computer scientist.",
                    temperature=0,
                    num_completions=4,
                    max_tokens=24,
                    top_k_per_token=3,
                    stop_sequences=["\n"],
                    echo_prompt=True,
                    top_p=0.3,
                ),
                {
                    "best_of": 3,
                    "echo": True,
                    "logprobs": 3,
                    "max_tokens": 24,
                    "model": "huggyllama/llama-7b",
                    "n": 4,
                    "prompt": "I am a computer scientist.",
                    "request_type": "language-model-inference",
                    "stop": ["\n"],
                    "temperature": 0,
                    "top_p": 0.3,
                },
            ),
            (
                Request(
                    model="stanford/alpaca-7b",
                    stop_sequences=["\n"],
                ),
                {
                    "best_of": 1,
                    "echo": False,
                    "logprobs": 1,
                    "max_tokens": 100,
                    "model": "togethercomputer/alpaca-7b",
                    "n": 1,
                    "prompt": "",
                    "request_type": "language-model-inference",
                    "stop": ["\n", "</s>"],
                    "temperature": 1.0,
                    "top_p": 1,
                },
            ),
            # TODO(#1828): Add test for `SET_DETAILS_TO_TRUE` after Together supports it.
        ],
    )
    def test_convert_to_raw_request(self, test_input, expected):
        assert expected == TogetherClient.convert_to_raw_request(test_input)

    def test_api_key_error(self):
        with pytest.raises(TogetherClientError):
            self.client.make_request(Request(model="together/bloom"))
