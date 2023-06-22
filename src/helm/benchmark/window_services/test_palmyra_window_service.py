import os
from tempfile import TemporaryDirectory
from typing import List

from .tokenizer_service import TokenizerService
from .window_service_factory import WindowServiceFactory
from .test_utils import get_tokenizer_service, TEST_PROMPT


class TestPalmyraWindowService:
    TEST_PROMPT_LENGTH: int = 61
    TEST_TOKEN_IDS: List[int] = [
        450,
        7817,
        363,
        10550,
        373,
        10606,
        3382,
        1379,
        313,
        11341,
        22192,
        29897,
        338,
        385,
        1006,
        2218,
        13326,
        3821,
        14511,
        1230,
        6345,
        714,
        310,
        278,
        7813,
        4006,
        8907,
        363,
        12968,
        29899,
        13409,
        287,
        3012,
        928,
        616,
        3159,
        28286,
        313,
        15715,
        29902,
        29897,
        393,
        263,
        9893,
        304,
        1207,
        15281,
        3061,
        2925,
        297,
        278,
        6559,
        29892,
        5849,
        29892,
        322,
        18209,
        310,
        22778,
        4733,
        29889,
    ]
    TEST_TOKENS: List[str] = [
        " The",
        " Center",
        " for",
        " Research",
        " on",
        " Foundation",
        " Mod",
        "els",
        " (",
        "CR",
        "FM",
        ")",
        " is",
        " an",
        " inter",
        "dis",
        "cipl",
        "inary",
        " initi",
        "ative",
        " born",
        " out",
        " of",
        " the",
        " Stan",
        "ford",
        " Institute",
        " for",
        " Human",
        "-",
        "Center",
        "ed",
        " Art",
        "ific",
        "ial",
        " Int",
        "elligence",
        " (",
        "HA",
        "I",
        ")",
        " that",
        " a",
        "ims",
        " to",
        " make",
        " fundamental",
        " adv",
        "ances",
        " in",
        " the",
        " study",
        ",",
        " development",
        ",",
        " and",
        " deployment",
        " of",
        " foundation",
        " models",
        ".",
    ]

    def setup_method(self):
        self.temporary_directory = TemporaryDirectory()
        with open(os.path.join(self.temporary_directory.name, "credentials.conf"), "w") as f:
            f.write("writerApiKey: secret")

        service: TokenizerService = get_tokenizer_service(self.temporary_directory.name)
        self.window_service = WindowServiceFactory.get_window_service("writer/palmyra-large", service)

    def teardown_method(self):
        self.temporary_directory.cleanup()

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 2000

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == self.TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == self.TEST_TOKENS

    def test_tokenize_and_count(self):
        assert self.window_service.get_num_tokens(TEST_PROMPT) == self.TEST_PROMPT_LENGTH

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - self.TEST_PROMPT_LENGTH
        )
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - self.TEST_PROMPT_LENGTH + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length
        long_prompt: str = TEST_PROMPT * 200
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)
