from helm.benchmark.scenarios.arxivrollbench_scenario import (
    ArxivRollBenchScenario,
    _dataset_path,
    _record_to_instance,
    _selection_to_letter,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


def test_dataset_path_compact_and_full():
    assert _dataset_path("2026a", "cs", "s", "compact") == "liangzid/robench2026a_test_all_category_setcsSCP-s-50"
    assert _dataset_path("2024b", "q-fin", "p", "full") == "liangzid/robench2024b_all_setq-finSCP-p"


def test_selection_to_letter():
    assert _selection_to_letter("Selection 1") == "A"
    assert _selection_to_letter("selection 4") == "D"
    assert _selection_to_letter("A") == "A"
    assert _selection_to_letter("1") == "1"


def test_domain_aliases():
    scenario = ArxivRollBenchScenario(release="2026a", domain="q-bio", task_type="s")

    assert scenario.domain == "q_bio"


def test_record_to_instance_prediction():
    record = {
        "context": "The introduction describes a new method.",
        "A": "A candidate",
        "B": "B candidate",
        "C": "C candidate",
        "D": "D candidate",
        "label": "C",
    }

    instance = _record_to_instance(record, "2026a", "cs", "p")

    assert instance.split == TEST_SPLIT
    assert "Context:\nThe introduction describes a new method." in instance.input.text
    assert [reference.tags for reference in instance.references] == [[], [], [CORRECT_TAG], []]
    assert instance.extra_data["task_type_name"] == "prediction"


def test_record_to_instance_selection():
    record = {
        "shuffled_text": "Paragraph with a blank.",
        "A": "A candidate",
        "B": "B candidate",
        "C": "C candidate",
        "D": "D candidate",
        "label": "Selection 2",
    }

    instance = _record_to_instance(record, "2026a", "math", "s")

    assert [reference.tags for reference in instance.references] == [[], [CORRECT_TAG], [], []]
    assert instance.extra_data["task_type_name"] == "sequencing"
