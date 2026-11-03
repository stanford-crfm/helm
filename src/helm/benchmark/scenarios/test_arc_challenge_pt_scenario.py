from collections import Counter
from unittest.mock import patch

import pytest

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import get_run_spec_function
from helm.benchmark.scenarios.arc_challenge_pt_scenario import ARC_Challenge_PT_Scenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Output, Reference, create_scenario


@pytest.fixture
def example():
    return {
        "id": "ARC-Challenge/test/Mercury_SC_415719",
        "answer": "A",
        "instruction": "Anna está segurando um cubo de gelo. Por que o cubo de gelo derrete em sua mão?",
        "option_a": "O calor se move de sua mão para o cubo de gelo.",
        "option_b": "O frio se move de sua mão para o cubo de gelo.",
        "option_c": "O calor se move do cubo de gelo para sua mão.",
        "option_d": "O frio se move do cubo de gelo para sua mão.",
        "option_e": None,
    }


def test_arc_challenge_pt_loads_portuguese_test_split(example, tmp_path):
    with patch(
        "helm.benchmark.scenarios.arc_challenge_pt_scenario.load_dataset", return_value=[example]
    ) as load_dataset:
        instances = ARC_Challenge_PT_Scenario().get_instances(str(tmp_path))

    load_dataset.assert_called_once_with("alexandrainst/m_arc", "pt", split="test", cache_dir=str(tmp_path / "data"))
    assert len(instances) == 1
    assert instances[0].input.text == example["instruction"]
    assert instances[0].split == TEST_SPLIT
    assert instances[0].references == [
        Reference(output=Output(text=example[f"option_{label}"]), tags=[CORRECT_TAG] if label == "a" else [])
        for label in "abcd"
    ]


@pytest.mark.parametrize(
    "options,answer,expected_choices,correct_index",
    [
        ({"option_d": None}, "C", "abc", 2),
        ({"option_e": "Uma quinta alternativa."}, "E", "abcde", 4),
        ({"option_b": None}, "D", "acd", 2),
        ({"option_b": "O calor se move de sua mão para o cubo de gelo."}, "B", "abcd", 1),
    ],
)
def test_arc_challenge_pt_matches_original_choice_labels(
    example, tmp_path, options, answer, expected_choices, correct_index
):
    example.update(options, answer=answer)
    # Exercise both absent columns and null values returned by Hugging Face.
    example = {key: value for key, value in example.items() if value is not None}
    with patch("helm.benchmark.scenarios.arc_challenge_pt_scenario.load_dataset", return_value=[example]):
        instance = ARC_Challenge_PT_Scenario().get_instances(str(tmp_path))[0]

    assert [ref.output.text for ref in instance.references] == [
        example[f"option_{label}"] for label in expected_choices
    ]
    assert [index for index, ref in enumerate(instance.references) if ref.is_correct] == [correct_index]


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"instruction": " "}, "empty or invalid question"),
        ({"instruction": None}, "empty or invalid question"),
        ({"option_a": ""}, "empty or invalid alternative A"),
        ({"option_b": 123}, "empty or invalid alternative B"),
        ({"answer": "E"}, "invalid answer label"),
        ({"answer": ""}, "invalid answer label"),
        ({"answer": None}, "invalid answer label"),
        ({"answer": 0}, "invalid answer label"),
        ({"option_a": None}, "invalid answer label"),
        ({"option_b": None, "option_c": None, "option_d": None}, "expected at least two alternatives"),
    ],
)
def test_arc_challenge_pt_rejects_invalid_examples(example, tmp_path, changes, message):
    example.update(changes)
    with patch("helm.benchmark.scenarios.arc_challenge_pt_scenario.load_dataset", return_value=[example]):
        with pytest.raises(ValueError, match=f"{example['id']}: {message}"):
            ARC_Challenge_PT_Scenario().get_instances(str(tmp_path))


def test_arc_challenge_pt_run_spec_is_discoverable():
    spec_function = get_run_spec_function("arc_challenge_pt")
    assert spec_function is not None
    spec = spec_function()
    assert isinstance(create_scenario(spec.scenario_spec), ARC_Challenge_PT_Scenario)
    assert spec.adapter_spec.method == ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL
    assert spec.adapter_spec.instructions == ""
    assert spec.adapter_spec.input_prefix == spec.adapter_spec.input_suffix == ""
    assert spec.adapter_spec.output_prefix == " "
    assert spec.adapter_spec.output_suffix == ""
    assert spec.adapter_spec.max_tokens == spec.adapter_spec.max_train_instances == 0
    assert spec.metric_specs == get_exact_match_metric_specs()
    assert spec.groups == ["arc_challenge_pt"]


@pytest.mark.scenarios
def test_arc_challenge_pt_dataset(example, tmp_path):
    instances = ARC_Challenge_PT_Scenario().get_instances(str(tmp_path))
    assert len(instances) == 1170
    assert Counter(len(instance.references) for instance in instances) == {3: 4, 4: 1163, 5: 3}
    for instance in instances:
        assert instance.split == TEST_SPLIT
        assert instance.input.text.strip()
        assert all(reference.output.text.strip() for reference in instance.references)
        assert sum(reference.is_correct for reference in instance.references) == 1

    matching_instances = [instance for instance in instances if instance.input.text == example["instruction"]]
    assert len(matching_instances) == 1
    assert matching_instances[0].references == [
        Reference(output=Output(text=example[f"option_{label}"]), tags=[CORRECT_TAG] if label == "a" else [])
        for label in "abcd"
    ]
