import os
import pytest
import tempfile

from helm.common.cache import BlackHoleCacheConfig, SqliteCacheConfig
from helm.common.request import Request
from helm.clients.openrouter_client import OpenRouterClient

from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer


class TestOpenRouterClient:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.tokenizer_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = HuggingFaceTokenizer(
            cache_config=BlackHoleCacheConfig(),
            tokenizer_name=self.tokenizer_name,
        )

    def teardown_method(self, method):
        os.remove(self.cache_path)

    @pytest.mark.parametrize(
        "model_name,test_input,expected_model",
        [
            (
                "mistralai/mistral-medium-3.1",
                Request(
                    model="mistralai/mistral-medium-3.1",
                    model_deployment="openrouter/mistral-medium-3.1",
                ),
                "mistralai/mistral-medium-3.1",
            ),
            (
                None,
                Request(model="openai/gpt-oss-20b:free", model_deployment="openrouter/gpt-oss-20b:free"),
                "openai/gpt-oss-20b:free",
            ),
        ],
    )
    def test_get_model_for_request(self, model_name, test_input, expected_model):
        client = OpenRouterClient(
            tokenizer_name=self.tokenizer_name,
            tokenizer=self.tokenizer,
            cache_config=SqliteCacheConfig(self.cache_path),
            model_name=model_name,
            api_key="test_key",
        )
        assert client._get_model_for_request(test_input) == expected_model

    def test_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        client = OpenRouterClient(
            tokenizer_name=self.tokenizer_name,
            tokenizer=self.tokenizer,
            cache_config=SqliteCacheConfig(self.cache_path),
        )
        assert client.api_key == "test_key"

    def test_api_key_argument(self):
        client = OpenRouterClient(
            tokenizer_name=self.tokenizer_name,
            tokenizer=self.tokenizer,
            cache_config=BlackHoleCacheConfig(),
            api_key="explicit_key",
        )
        assert client.api_key == "explicit_key"
