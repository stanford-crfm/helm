import json
import os
from tempfile import TemporaryDirectory
from typing import List, Set

import pytest

from helm.benchmark.scenarios.med_dialog_scenario import MedDialogScenario


def _seed_split_file(output_path: str, subset: str, examples: List) -> None:
    """Pre-create the test split file so get_instances does not download."""
    data_path = os.path.join(output_path, subset)
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, "test.json"), "w") as f:
        json.dump({"data": examples}, f)


def test_med_dialog_scenario_dedup():
    examples = [
        {"src": "Patient: A", "tgt": "Summary A"},
        {"src": "Patient: B", "tgt": "Summary B"},
        {"src": "Patient: A", "tgt": "Summary A duplicate"},
        {"src": "Patient: C", "tgt": "Summary C"},
        {"src": "Patient: B", "tgt": "Summary B duplicate"},
    ]
    with TemporaryDirectory() as tmpdir:
        _seed_split_file(tmpdir, "icliniq", examples)
        instances = MedDialogScenario(subset="icliniq").get_instances(tmpdir)
        deduped_instances = MedDialogScenario(subset="icliniq", dedup=True).get_instances(tmpdir)

    # Without dedup, the scenario is unchanged: every entry is kept, duplicates included.
    assert [instance.input.text for instance in instances] == [example["src"] for example in examples]

    # With dedup, each source text appears exactly once, keeping the first
    # occurrence in the original order.
    assert [instance.input.text for instance in deduped_instances] == ["Patient: A", "Patient: B", "Patient: C"]
    assert [instance.references[0].output.text for instance in deduped_instances] == [
        "Summary A",
        "Summary B",
        "Summary C",
    ]


@pytest.mark.scenarios
def test_med_dialog_scenario_dedup_real_data():
    with TemporaryDirectory() as tmpdir:
        instances = MedDialogScenario(subset="icliniq").get_instances(tmpdir)
        deduped_instances = MedDialogScenario(subset="icliniq", dedup=True).get_instances(tmpdir)

    # Without dedup, the scenario is unchanged: it keeps every entry, duplicates included.
    assert len(instances) == 3108
    source_texts = [instance.input.text for instance in instances]
    assert len(set(source_texts)) == 2069

    # With dedup, each source text appears exactly once.
    deduped_texts = [instance.input.text for instance in deduped_instances]
    assert len(deduped_texts) == len(set(deduped_texts)) == 2069

    # Dedup keeps the first occurrence of each source text, in the original order.
    seen: Set[str] = set()
    first_occurrences: List[str] = []
    for text in source_texts:
        if text not in seen:
            seen.add(text)
            first_occurrences.append(text)
    assert deduped_texts == first_occurrences
