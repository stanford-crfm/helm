from typing import List

from yalm_tokenizer import YaLMTokenizer


class TestYaLMTokenizer:
    def setup_method(self):
        self.tokenizer = YaLMTokenizer()

    def test_tokenize(self):
        text: str = "hello world!"  # Has 3 tokens
        token_ids: List[int] = self.tokenizer.tokenize(text)
        assert token_ids == [49524, 2175, 127679]
        assert self.tokenizer.convert_ids_to_tokens(token_ids) == ["▁hello", "▁world", "!"]

    def test_decode(self):
        text: str = "should be exactly the same"
        assert self.tokenizer.decode(self.tokenizer.tokenize(text)) == text
