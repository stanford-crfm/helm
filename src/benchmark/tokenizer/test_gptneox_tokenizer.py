import shutil
import tempfile
from typing import List

from .tokenizer_service import TokenizerService
from .tokenizer_factory import TokenizerFactory
from .test_gpt2_tokenizer import TEST_PROMPT
from .test_utils import get_tokenizer_service


class TestGPTNeoXTokenizer:
    TEST_TOKEN_IDS: List[int] = [
        510,
        5197,
        323,
        5489,
        327,
        6807,
        31581,
        313,
        1311,
        18278,
        10,
        310,
        271,
        734,
        36078,
        15952,
        5686,
        562,
        273,
        253,
        22681,
        6516,
        323,
        8801,
        14,
        18760,
        2122,
        3975,
        11232,
        19256,
        313,
        10114,
        42,
        10,
        326,
        13698,
        281,
        1056,
        7936,
        16424,
        275,
        253,
        1263,
        13,
        2440,
        13,
        285,
        19007,
        273,
        12153,
        3210,
        15,
    ]

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path)
        self.tokenizer = TokenizerFactory.get_tokenizer("together/gpt-neox-20b", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_sequence_length(self):
        assert self.tokenizer.max_sequence_length == 2048

    def test_max_request_length(self):
        assert self.tokenizer.max_request_length == 2049

    def test_tokenizer_name(self):
        assert self.tokenizer.tokenizer_name == "huggingface/gpt-neox-20b"

    def test_encode(self):
        assert self.tokenizer.encode(TEST_PROMPT).tokens == TestGPTNeoXTokenizer.TEST_TOKEN_IDS

    def test_decode(self):
        assert self.tokenizer.decode(TestGPTNeoXTokenizer.TEST_TOKEN_IDS) == TEST_PROMPT

    def test_tokenize(self):
        assert self.tokenizer.tokenize(TEST_PROMPT) == [
            "The",
            "ĠCenter",
            "Ġfor",
            "ĠResearch",
            "Ġon",
            "ĠFoundation",
            "ĠModels",
            "Ġ(",
            "CR",
            "FM",
            ")",
            "Ġis",
            "Ġan",
            "Ġinter",
            "disciplinary",
            "Ġinitiative",
            "Ġborn",
            "Ġout",
            "Ġof",
            "Ġthe",
            "ĠStanford",
            "ĠInstitute",
            "Ġfor",
            "ĠHuman",
            "-",
            "Cent",
            "ered",
            "ĠArt",
            "ificial",
            "ĠIntelligence",
            "Ġ(",
            "HA",
            "I",
            ")",
            "Ġthat",
            "Ġaims",
            "Ġto",
            "Ġmake",
            "Ġfundamental",
            "Ġadvances",
            "Ġin",
            "Ġthe",
            "Ġstudy",
            ",",
            "Ġdevelopment",
            ",",
            "Ġand",
            "Ġdeployment",
            "Ġof",
            "Ġfoundation",
            "Ġmodels",
            ".",
        ]

    def test_tokenize_and_count(self):
        # There are 52 tokens in `TEST_PROMPT`.
        assert self.tokenizer.tokenize_and_count(TEST_PROMPT) == 52

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.tokenizer.fits_within_context_window(TEST_PROMPT, 2049 - 52)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.tokenizer.fits_within_context_window(TEST_PROMPT, 2049 - 52 + 1)

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 51 * 41 = 2091 tokens
        long_prompt: str = TEST_PROMPT * 41
        assert not self.tokenizer.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.tokenizer.truncate_from_right(long_prompt)
        assert self.tokenizer.tokenize_and_count(truncated_long_prompt) == 2049
        assert self.tokenizer.fits_within_context_window(truncated_long_prompt)
