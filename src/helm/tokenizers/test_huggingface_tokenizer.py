import os
import tempfile
from typing import Optional

from helm.common.cache import SqliteCacheConfig
from helm.common.general import parallel_map, singleton
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer


class TestHuggingFaceGPT2Tokenizer:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name
        self.tokenizer = HuggingFaceTokenizer(
            SqliteCacheConfig(self.cache_path),
            tokenizer_name="huggingface/gpt2",
            pretrained_model_name_or_path="openai-community/gpt2",
        )

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_tokenize(self):
        request = TokenizationRequest(text="I am a computer scientist.", tokenizer="huggingface/gpt2")
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        result = self.tokenizer.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == ["I", " am", " a", " computer", " scientist", "."]

    def test_encode(self):
        request = TokenizationRequest(
            text="I am a computer scientist.", tokenizer="huggingface/gpt2", encode=True, truncation=True, max_length=1
        )
        result: TokenizationRequestResult = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making the tokenize request. Result should not be cached"
        result = self.tokenizer.tokenize(request)
        assert result.cached, "Result should be cached"
        assert result.raw_tokens == [40]

        request = TokenizationRequest(
            text="I am a computer scientist.",
            tokenizer="huggingface/gpt2",
            encode=True,
            truncation=True,
            max_length=1024,
        )
        result = self.tokenizer.tokenize(request)
        assert not result.cached, "First time making this particular request. Result should not be cached"
        assert result.raw_tokens == [40, 716, 257, 3644, 11444, 13]

    def test_decode(self):
        request = DecodeRequest(tokens=[40, 716, 257, 3644, 11444, 13], tokenizer="huggingface/gpt2")
        result: DecodeRequestResult = self.tokenizer.decode(request)
        assert not result.cached, "First time making the decode request. Result should not be cached"
        result = self.tokenizer.decode(request)
        assert result.cached, "Result should be cached"
        assert result.text == "I am a computer scientist."

    def test_already_borrowed(self):
        """Test workaround of the "Already borrowed" bug (#1421) caused by the thread-hostile Hugging Face tokenizer"""

        def make_tokenize_request(seed: int) -> None:
            request_length = 10
            truncation = bool(seed % 2)
            self.tokenizer.tokenize(
                # The truncation parameter requires setting a flag on the Rust FastTokenizer.
                # Concurrent requests cause concurrent mutations, which results an Rust concurrency error.
                TokenizationRequest(
                    text=str(seed) * request_length, tokenizer="huggingface/gpt2", encode=True, truncation=truncation
                )
            )

        num_requests = 100
        # Should not raise "Already borrowed" error
        parallel_map(make_tokenize_request, list(range(num_requests)), parallelism=8)


class TestHuggingFaceTokenizer:
    # The following prompt has 51 tokens according to the GPT-2 tokenizer
    TEST_PROMPT: str = (
        "The Center for Research on Foundation Models (CRFM) is "
        "an interdisciplinary initiative born out of the Stanford "
        "Institute for Human-Centered Artificial Intelligence (HAI) "
        "that aims to make fundamental advances in the study, development, "
        "and deployment of foundation models."
    )

    @staticmethod
    def verify_get_tokenizer(
        tokenizer_name: str, expected_num_tokens: int, pretrained_model_name_or_path: Optional[str] = None
    ):
        wrapped_tokenizer = HuggingFaceTokenizer.get_tokenizer(
            helm_tokenizer_name=tokenizer_name,
            pretrained_model_name_or_path=pretrained_model_name_or_path or tokenizer_name,
        )
        assert tokenizer_name in HuggingFaceTokenizer._tokenizers, "Tokenizer should be cached"
        with wrapped_tokenizer as tokenizer:
            assert len(tokenizer.encode(TestHuggingFaceTokenizer.TEST_PROMPT)) == expected_num_tokens

    def test_get_tokenizer_gpt2(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("huggingface/gpt2", 51, pretrained_model_name_or_path="gpt2")

    def test_get_tokenizer_gptj(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("EleutherAI/gpt-j-6B", 51)

    def test_get_tokenizer_gptneox(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("EleutherAI/gpt-neox-20b", 52)

    def test_get_tokenizer_bloom(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("bigscience/bloom", 51)

    def test_get_tokenizer_t0pp(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("bigscience/T0pp", 58)

    def test_get_tokenizer_t511b(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("google/t5-11b", 58, pretrained_model_name_or_path="t5-11b")

    def test_get_tokenizer_ul2(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("google/ul2", 58)

    def test_get_santacoder(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("bigcode/santacoder", 62)

    def test_get_clip_tokenizer(self):
        TestHuggingFaceTokenizer.verify_get_tokenizer("openai/clip-vit-large-patch14", 50)

    def test_gpt2_tokenize_eos(self):
        eos_token: str = "<|endoftext|>"
        wrapped_tokenizer = HuggingFaceTokenizer.get_tokenizer("huggingface/gpt2", pretrained_model_name_or_path="gpt2")
        with wrapped_tokenizer as tokenizer:
            token_ids = tokenizer.encode(eos_token)
            assert singleton(token_ids) == 50256
            assert tokenizer.decode(token_ids) == eos_token
