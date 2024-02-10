# mypy: check_untyped_defs = False
from typing import List, DefaultDict, Set

from collections import defaultdict
from compute_data_overlap_metrics import (
    compute_document_data_overlap,
    create_ngram_index,
    EntryDataOverlapKey,
    Ngram,
    NgramIndex,
    PART_INPUT,
    PART_REF,
)
from data_overlap_spec import DataOverlapStatsKey, OverlapProtocolSpec
from light_scenario import LightScenario, LightInstance, LightScenarioKey
from light_tokenizer import LightTokenizer, DefaultTokenizer
from scenarios.scenario import ScenarioSpec

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
    scenario_key=LightScenarioKey(
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario", args={}
        ),
        split="test",
    ),
    instances=[
        LightInstance(input="Center for Research on Foundation", references=["bar", "baz"], id="id1"),
        LightInstance(input="bar bar", references=["foo", "baz"], id="id2"),
    ],
)
TEST_SCENARIO_2 = LightScenario(
    scenario_key=LightScenarioKey(
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario2", args={}
        ),
        split="test",
    ),
    instances=[LightInstance(input=TEST_DOCUMENT, references=[TEST_DOCUMENT, TEST_DOCUMENT], id="id1")],
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


def test_create_ngram_index():
    tokenizer = LightTokenizer()
    stats_key_counts = dict()
    scenarios = [TEST_SCENARIO_1, TEST_SCENARIO_2]
    ngram_index: NgramIndex
    ngram_index = create_ngram_index(
        light_scenarios=scenarios, n_values=N_VALUES, tokenizer=tokenizer, stats_key_counts=stats_key_counts
    )

    stats_1_key, stats_2_key, stats_3_key = (
        DataOverlapStatsKey(
            light_scenario_key=TEST_SCENARIO_1.scenario_key, overlap_protocol_spec=OverlapProtocolSpec(n=5)
        ),
        DataOverlapStatsKey(
            light_scenario_key=TEST_SCENARIO_2.scenario_key, overlap_protocol_spec=OverlapProtocolSpec(n=5)
        ),
        DataOverlapStatsKey(
            light_scenario_key=TEST_SCENARIO_2.scenario_key, overlap_protocol_spec=OverlapProtocolSpec(n=13)
        ),
    )

    test_5_gram: Ngram = ("Center", "for", "Research", "on", "Foundation")
    test_13_gram: Ngram = (
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
        [
            EntryDataOverlapKey(stats_key=stats_1_key, instance_id="id1", part=PART_INPUT),
            EntryDataOverlapKey(stats_key=stats_2_key, instance_id="id1", part=PART_INPUT),
            EntryDataOverlapKey(stats_key=stats_2_key, instance_id="id1", part=PART_REF),
        ]
    )
    assert ngram_index[13][test_13_gram] == set(
        [
            EntryDataOverlapKey(stats_key=stats_3_key, instance_id="id1", part=PART_INPUT),
            EntryDataOverlapKey(stats_key=stats_3_key, instance_id="id1", part=PART_REF),
        ]
    )


def test_compute_document_data_overlap():
    tokenizer = LightTokenizer()
    stats_key_counts = dict()
    scenarios = [TEST_SCENARIO_1, TEST_SCENARIO_2]
    ngram_index: NgramIndex
    ngram_index = create_ngram_index(
        light_scenarios=scenarios, n_values=N_VALUES, tokenizer=tokenizer, stats_key_counts=stats_key_counts
    )

    stats_key_to_input_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)
    stats_key_to_reference_ids: DefaultDict[DataOverlapStatsKey, Set] = defaultdict(set)

    compute_document_data_overlap(
        document=TEST_DOCUMENT,
        ngram_index=ngram_index,
        tokenizer=tokenizer,
        stats_key_to_input_ids=stats_key_to_input_ids,
        stats_key_to_reference_ids=stats_key_to_reference_ids,
        entry_overlap_key_to_ngram_counts=None,
        output_ngrams=False,
    )
    assert stats_key_to_input_ids == defaultdict(
        set,
        {
            DataOverlapStatsKey(
                light_scenario_key=LightScenarioKey(
                    scenario_spec=ScenarioSpec(
                        class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario2", args={}
                    ),
                    split="test",
                ),
                overlap_protocol_spec=OverlapProtocolSpec(n=5),
            ): {"id1"},
            DataOverlapStatsKey(
                light_scenario_key=LightScenarioKey(
                    scenario_spec=ScenarioSpec(
                        class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario", args={}
                    ),
                    split="test",
                ),
                overlap_protocol_spec=OverlapProtocolSpec(n=5),
            ): {"id1"},
            DataOverlapStatsKey(
                light_scenario_key=LightScenarioKey(
                    scenario_spec=ScenarioSpec(
                        class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario2", args={}
                    ),
                    split="test",
                ),
                overlap_protocol_spec=OverlapProtocolSpec(n=13),
            ): {"id1"},
        },
    )

    assert stats_key_to_reference_ids == defaultdict(
        set,
        {
            DataOverlapStatsKey(
                light_scenario_key=LightScenarioKey(
                    scenario_spec=ScenarioSpec(
                        class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario2", args={}
                    ),
                    split="test",
                ),
                overlap_protocol_spec=OverlapProtocolSpec(n=5),
            ): {"id1"},
            DataOverlapStatsKey(
                light_scenario_key=LightScenarioKey(
                    scenario_spec=ScenarioSpec(
                        class_name="helm.benchmark.scenarios.natural_qa_scenario.NaturalQAScenario2", args={}
                    ),
                    split="test",
                ),
                overlap_protocol_spec=OverlapProtocolSpec(n=13),
            ): {"id1"},
        },
    )
