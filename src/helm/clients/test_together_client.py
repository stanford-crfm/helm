import os
import pytest
import tempfile

from helm.common.cache import BlackHoleCacheConfig, SqliteCacheConfig
from helm.common.request import Request

from helm.clients.together_client import (
    TogetherClient,
    TogetherChatClient,
    TogetherCompletionClient,
    TogetherClientError,
)


class TestTogetherClient:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name

    def teardown_method(self, method):
        os.remove(self.cache_path)

    @pytest.mark.parametrize(
        "together_model,test_input,expected",
        [
            (
                "togethercomputer/RedPajama-INCITE-Base-3B-v1",
                Request(
                    model="together/redpajama-incite-base-3b-v1",
                    model_deployment="together/redpajama-incite-base-3b-v1",
                ),
                {
                    "best_of": 1,
                    "echo": False,
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
                "huggyllama/llama-7b",
                Request(
                    model="meta/llama-7b",
                    model_deployment="together/llama-7b",
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
                "togethercomputer/alpaca-7b",
                Request(
                    model="stanford/alpaca-7b",
                    model_deployment="together/alpaca-7b",
                    stop_sequences=["\n"],
                ),
                {
                    "best_of": 1,
                    "echo": False,
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
    def test_convert_to_raw_request(self, together_model, test_input, expected):
        client = TogetherClient(
            cache_config=SqliteCacheConfig(self.cache_path),
            together_model=together_model,
        )
        assert expected == client.convert_to_raw_request(test_input)

    def test_api_key_error(self):
        client = TogetherClient(
            cache_config=SqliteCacheConfig(self.cache_path),
            together_model="togethercomputer/RedPajama-INCITE-Base-3B-v1",
        )
        with pytest.raises(TogetherClientError):
            client.make_request(
                Request(
                    model="together/redpajama-incite-base-3b-v1",
                    model_deployment="together/redpajama-incite-base-3b-v1",
                )
            )


@pytest.mark.models
def test_together_chat_client_make_request():
    # Requires setting TOGETHER_API_KEY environment variable.
    client = TogetherChatClient(
        cache_config=BlackHoleCacheConfig(), api_key=None, together_model="meta-llama/Llama-3-8b-chat-hf"
    )
    request = Request(
        model="meta/llama-3-8b-instruct",
        model_deployment="together/llama-3-8b-instruct",
        prompt="Elephants are one of the most",
        temperature=0.0,
        max_tokens=10,
    )
    result = client.make_request(request)
    assert result.success
    assert not result.cached
    assert result.embedding == []
    assert len(result.completions) == 1
    assert result.completions[0].text == "...intelligent animals on Earth!assistant"
    assert result.completions[0].logprob == 0.0
    result_token_strings = [token.text for token in result.completions[0].tokens]
    assert result_token_strings == [
        "...",
        "int",
        "elligent",
        " animals",
        " on",
        " Earth",
        "!",
        "<|eot_id|>",
        "<|start_header_id|>",
        "assistant",
    ]


@pytest.mark.models
def test_together_completion_client_make_request():
    # Requires setting TOGETHER_API_KEY environment variable.
    client = TogetherCompletionClient(
        cache_config=BlackHoleCacheConfig(), api_key=None, together_model="meta-llama/Llama-3-8b-hf"
    )
    request = Request(
        model="meta/llama-3-8b",
        model_deployment="together/llama-3-8b",
        prompt="Elephants are one of the most",
        temperature=0.0,
        max_tokens=10,
    )
    result = client.make_request(request)
    assert result.success
    assert not result.cached
    assert result.embedding == []
    assert len(result.completions) == 1
    assert result.completions[0].text == " popular animals in the world. They are known for"
    assert result.completions[0].logprob == 0.0
    result_token_strings = [token.text for token in result.completions[0].tokens]
    assert result_token_strings == [
        " popular",
        " animals",
        " in",
        " the",
        " world",
        ".",
        " They",
        " are",
        " known",
        " for",
    ]
