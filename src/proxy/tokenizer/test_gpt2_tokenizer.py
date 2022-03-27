from typing import List

from .gpt2_tokenizer import GPT2Tokenizer

TEST_PROMPT: str = (
    "The Center for Research on Foundation Models (CRFM) is "
    "an interdisciplinary initiative born out of the Stanford "
    "Institute for Human-Centered Artificial Intelligence (HAI) "
    "that aims to make fundamental advances in the study, development, "
    "and deployment of foundation models."
)

TEST_TOKEN_IDS: List[int] = [
    464,
    3337,
    329,
    4992,
    319,
    5693,
    32329,
    357,
    9419,
    23264,
    8,
    318,
    281,
    987,
    40625,
    10219,
    4642,
    503,
    286,
    262,
    13863,
    5136,
    329,
    5524,
    12,
    19085,
    1068,
    35941,
    9345,
    357,
    7801,
    40,
    8,
    326,
    12031,
    284,
    787,
    7531,
    14901,
    287,
    262,
    2050,
    11,
    2478,
    11,
    290,
    14833,
    286,
    8489,
    4981,
    13,
]

TEST_TOKENS: List[str] = [
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
    "ĠArtificial",
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


class TestGPT2Tokenizer:
    def setup_method(self):
        self.tokenizer = GPT2Tokenizer()

    def test_encode(self):
        assert self.tokenizer.encode(TEST_PROMPT) == TEST_TOKEN_IDS

    def test_decode(self):
        assert self.tokenizer.decode(TEST_TOKEN_IDS) == TEST_PROMPT

    def test_tokenize(self):
        assert self.tokenizer.tokenize(TEST_PROMPT) == TEST_TOKENS

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.tokenizer.fits_within_context_window(TEST_PROMPT, 1024 - 51)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.tokenizer.fits_within_context_window(TEST_PROMPT, 1024 - 51 + 1)

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 51 * 41 = 2091 tokens
        long_prompt: str = TEST_PROMPT * 41
        assert not self.tokenizer.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.tokenizer.truncate_from_right(long_prompt)
        assert self.tokenizer.tokenize_and_count(truncated_long_prompt) == 1024
        assert self.tokenizer.fits_within_context_window(truncated_long_prompt)

    def test_tokenize_and_count(self):
        assert self.tokenizer.tokenize_and_count(TEST_PROMPT) == 51
