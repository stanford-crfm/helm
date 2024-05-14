import pytest

from helm.common.cache import BlackHoleCacheConfig
from helm.common.request import Request, RequestResult
from helm.clients.huggingface_client import HuggingFaceClient
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer


class TestHuggingFaceClient:
    def test_gpt2(self):
        tokenizer = HuggingFaceTokenizer(
            BlackHoleCacheConfig(), "huggingface/gpt2", pretrained_model_name_or_path="openai/gpt2"
        )
        client = HuggingFaceClient(
            cache_config=BlackHoleCacheConfig(),
            tokenizer=tokenizer,
            pretrained_model_name_or_path="openai-community/gpt2",
        )
        prompt: str = "I am a computer scientist."
        result: RequestResult = client.make_request(
            Request(
                model="openai/gpt2",
                model_deployment="huggingface/gpt2",
                prompt=prompt,
                num_completions=3,
                top_k_per_token=5,
                max_tokens=1,
                echo_prompt=True,
            )
        )
        assert len(result.completions) == 3
        assert result.completions[0].text.startswith(
            prompt
        ), "echo_prompt was set to true. Expected the prompt at the beginning of each completion"

    @pytest.mark.skip(reason="GPT-J 6B is 22 GB and extremely slow without a GPU.")
    def test_gptj_6b(self):
        tokenizer = HuggingFaceTokenizer(
            BlackHoleCacheConfig(), "huggingface/gpt2", pretrained_model_name_or_path="openai/gpt2"
        )
        client = HuggingFaceClient(
            cache_config=BlackHoleCacheConfig(),
            tokenizer=tokenizer,
            pretrained_model_name_or_path="openai-community/gpt2",
        )
        result: RequestResult = client.make_request(
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
        tokenizer = HuggingFaceTokenizer(
            BlackHoleCacheConfig(), "huggingface/gpt2", pretrained_model_name_or_path="openai/gpt2"
        )
        client = HuggingFaceClient(
            cache_config=BlackHoleCacheConfig(),
            tokenizer=tokenizer,
            pretrained_model_name_or_path="openai-community/gpt2",
        )
        prompt: str = "I am a computer scientist."
        result: RequestResult = client.make_request(
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
