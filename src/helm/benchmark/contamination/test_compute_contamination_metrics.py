from typing import Set, Dict, DefaultDict, Tuple, List

from helm.benchmark.contamination.compute_contamination_metrics import (
    compute_scenario_document_contamination,
    create_stats_and_ngram_index,
)
from helm.benchmark.contamination.light_scenario import LightScenario, LightInstance
from helm.benchmark.contamination.light_tokenizer import LightTokenizer, DefaultTokenizer
from helm.benchmark.contamination.contamination_stats import ContaminationStats, PART_INPUT, PART_REF

N_VALUES = [5, 13]

TEST_DOCUMENT: str = (
    "The Center for Research on Foundation Models (CRFM) is "
    "an interdisciplinary initiative born out of the Stanford "
    "Institute for Human-Centered Artificial Intelligence (HAI) "
    "that aims to make fundamental advances in the study, development, "
    "and deployment of foundation models."
)

TEST_TOKENS_SPLIT_BY_SPACE: List[str] = [
    "The",
    "Center",
    "for",
    "Research",
    "on",
    "Foundation",
    "Models",
    "(CRFM)",
    "is",
    "an",
    "interdisciplinary",
    "initiative",
    "born",
    "out",
    "of",
    "the",
    "Stanford",
    "Institute",
    "for",
    "Human-Centered",
    "Artificial",
    "Intelligence",
    "(HAI)",
    "that",
    "aims",
    "to",
    "make",
    "fundamental",
    "advances",
    "in",
    "the",
    "study,",
    "development,",
    "and",
    "deployment",
    "of",
    "foundation",
    "models.",
]

TEST_TOKENS_BY_DEFAULT_TOKENIZER: List[str] = [
    "the",
    "center",
    "for",
    "research",
    "on",
    "foundation",
    "models",
    "crfm",
    "is",
    "an",
    "interdisciplinary",
    "initiative",
    "born",
    "out",
    "of",
    "the",
    "stanford",
    "institute",
    "for",
    "human",
    "centered",
    "artificial",
    "intelligence",
    "hai",
    "that",
    "aims",
    "to",
    "make",
    "fundamental",
    "advances",
    "in",
    "the",
    "study",
    "development",
    "and",
    "deployment",
    "of",
    "foundation",
    "models",
    "",
]

TEST_SCENARIO_1 = LightScenario(
    light_scenario_spec={"name": "TEST_SCENARIO_1"},
    light_instances=[
        LightInstance(input="Center for Research on Foundation", references=["bar", "baz"]),
        LightInstance(input="bar bar", references=["foo", "baz"]),
    ],
)
TEST_SCENARIO_2 = LightScenario(
    light_scenario_spec={"name": "TEST_SCENARIO_2"},
    light_instances=[LightInstance(input=TEST_DOCUMENT, references=[TEST_DOCUMENT, TEST_DOCUMENT])],
)


def test_light_tokenizer():
    light_tokenizer = LightTokenizer()
    default_tokenizer = DefaultTokenizer()

    assert light_tokenizer.tokenize(TEST_DOCUMENT) == TEST_TOKENS_SPLIT_BY_SPACE
    assert default_tokenizer.tokenize(TEST_DOCUMENT) == TEST_TOKENS_BY_DEFAULT_TOKENIZER

    simple_tokenization_test_case: str = "THis ,,iS a SiMPlE t-EsT cAsE"
    assert light_tokenizer.tokenize(simple_tokenization_test_case) == ["THis", ",,iS", "a", "SiMPlE", "t-EsT", "cAsE"]
    assert default_tokenizer.tokenize(simple_tokenization_test_case) == [
        "this",
        "is",
        "a",
        "simple",
        "t",
        "est",
        "case",
    ]


def test_create_stats_and_ngram_index():
    tokenizer = LightTokenizer()
    scenarios = [TEST_SCENARIO_1, TEST_SCENARIO_2]
    all_contamination_stats: Dict[str, ContaminationStats]
    ngram_index: DefaultDict[int, DefaultDict[Tuple[str], Set[tuple]]]
    all_contamination_stats, ngram_index = create_stats_and_ngram_index(
        light_scenarios=scenarios, n_values=N_VALUES, tokenizer=tokenizer
    )

    stats_1_repr, stats_2_repr, stats_3_repr = (
        str({"light_scenario_spec": TEST_SCENARIO_1.light_scenario_spec, "N": 5}),
        str({"light_scenario_spec": TEST_SCENARIO_2.light_scenario_spec, "N": 5}),
        str({"light_scenario_spec": TEST_SCENARIO_2.light_scenario_spec, "N": 13}),
    )
    stats_1, stats_2, stats_3 = (
        all_contamination_stats[stats_1_repr],
        all_contamination_stats[stats_2_repr],
        all_contamination_stats[stats_3_repr],
    )
    assert stats_1.num_instances == 2 and stats_2.num_instances == 1 and stats_3.num_instances == 1

    test_5_gram: Tuple(str) = ("Center", "for", "Research", "on", "Foundation")
    test_13_gram: Tuple(str) = (
        "The",
        "Center",
        "for",
        "Research",
        "on",
        "Foundation",
        "Models",
        "(CRFM)",
        "is",
        "an",
        "interdisciplinary",
        "initiative",
        "born",
    )
    assert ngram_index[5][test_5_gram] == set(
        [(stats_1, 0, PART_INPUT), (stats_2, 0, PART_INPUT), (stats_2, 0, PART_REF)]
    )
    assert ngram_index[13][test_13_gram] == set([(stats_3, 0, PART_INPUT), (stats_3, 0, PART_REF)])


def test_compute_scenario_document_contamination():
    tokenizer = LightTokenizer()
    scenarios = [TEST_SCENARIO_1, TEST_SCENARIO_2]
    all_contamination_stats: Dict[str, ContaminationStats]
    ngram_index: DefaultDict[int, DefaultDict[Tuple[str], Set[tuple]]]
    all_contamination_stats, ngram_index = create_stats_and_ngram_index(
        light_scenarios=scenarios, n_values=N_VALUES, tokenizer=tokenizer
    )

    compute_scenario_document_contamination(ngram_index, document=TEST_DOCUMENT, n_values=N_VALUES, tokenizer=tokenizer)

    stats_1_repr, stats_2_repr, stats_3_repr = (
        str({"light_scenario_spec": TEST_SCENARIO_1.light_scenario_spec, "N": 5}),
        str({"light_scenario_spec": TEST_SCENARIO_2.light_scenario_spec, "N": 5}),
        str({"light_scenario_spec": TEST_SCENARIO_2.light_scenario_spec, "N": 13}),
    )
    stats_1, stats_2, stats_3 = (
        all_contamination_stats[stats_1_repr],
        all_contamination_stats[stats_2_repr],
        all_contamination_stats[stats_3_repr],
    )

    assert stats_1.dirty_input_fraction == 0.5 and stats_1.dirty_reference_fraction == 0
    assert stats_2.dirty_input_fraction == 1 and stats_2.dirty_reference_fraction == 1
    assert stats_3.dirty_input_fraction == 1 and stats_3.dirty_reference_fraction == 1
