from .tokenizer_factory import TokenizerFactory


class TestTokenizerFactory:
    def test_cache(self):
        # Verify the correct tokenizer is fetched
        tokenizer = TokenizerFactory.get_tokenizer(model="huggingface/gpt2")
        assert tokenizer.max_sequence_length == 1024

        # Check the tokenizer is cached
        assert "huggingface/gpt2" in TokenizerFactory.cached_tokenizers

    def test_no_tokenizer_service_for_ai21(self):
        # Verify that not passing a `TokenizerService` for AI21 throws an error
        try:
            TokenizerFactory.get_tokenizer(model="ai21/j1-large")
            assert False
        except Exception as e:
            assert isinstance(e, ValueError)
