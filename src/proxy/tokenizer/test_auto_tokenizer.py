from typing import List

from .auto_tokenizer import AutoTokenizer


class TestAutoTokenizer:
    """
    Use the OpenAITokenzier as an example to test AutoTokenizer.
    """

    MODEL: str = "openai/davinci"

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

    def setup_method(self):
        self.tokenizer = AutoTokenizer()

    def test_encode(self):
        assert (
            self.tokenizer.encode(TestAutoTokenizer.TEST_PROMPT, TestAutoTokenizer.MODEL)
            == TestAutoTokenizer.TEST_TOKEN_IDS
        )

    def test_decode(self):
        assert (
            self.tokenizer.decode(TestAutoTokenizer.TEST_TOKEN_IDS, TestAutoTokenizer.MODEL)
            == TestAutoTokenizer.TEST_PROMPT
        )

    def test_tokenize(self):
        assert (
            self.tokenizer.tokenize(TestAutoTokenizer.TEST_PROMPT, TestAutoTokenizer.MODEL)
            == TestAutoTokenizer.TEST_TOKENS
        )
