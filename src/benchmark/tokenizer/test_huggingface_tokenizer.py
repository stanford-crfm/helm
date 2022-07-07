from transformers import GPT2TokenizerFast

from .huggingface_tokenizer import HuggingFaceTokenizers


class TestHuggingFaceTokenizers:
    def test_get_tokenizer(self):
        tokenizer_name: str = "huggingface/gpt2-tokenizer-fast"
        tokenizer = HuggingFaceTokenizers.get_tokenizer(tokenizer_name)
        assert tokenizer_name in HuggingFaceTokenizers.tokenizers, "Tokenizer should be cached"
        assert type(tokenizer) == GPT2TokenizerFast
