import os
import re
from typing import List, Literal, Optional, Tuple

import datasets

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
    ScenarioMetadata,
)
from helm.common.general import ensure_directory_exists


DOMAINS: List[Tuple[str, str]] = [
    ("cs", "cs"),
    ("q_fin", "q-fin"),
    ("math", "math"),
    ("physics", "physics"),
    ("stat", "stat"),
    ("q_bio", "q-bio"),
    ("econ", "econ"),
    ("eess", "eess"),
]
DOMAIN_ALIASES = {
    "q-fin": "q_fin",
    "q-bio": "q_bio",
}
RELEASES: List[str] = ["2024b", "2025a", "2026a"]
TASK_TYPES: List[str] = ["s", "c", "p"]
TASK_TYPE_NAMES = {
    "s": "sequencing",
    "c": "cloze",
    "p": "prediction",
}


def _dataset_path(
    release: str,
    hf_domain: str,
    task_type: str,
    split: Literal["compact", "full"],
) -> str:
    suffix = "-50" if split == "compact" else ""
    if release == "2024b":
        return f"liangzid/robench2024b_all_set{hf_domain}SCP-{task_type}{suffix}"
    return f"liangzid/robench{release}_test_all_category_set" f"{hf_domain}SCP-{task_type}{suffix}"


def _selection_to_letter(label: str) -> str:
    match = re.search(r"\bselection\s*([1-4])\b", str(label), re.IGNORECASE)
    if match:
        return chr(ord("A") + int(match.group(1)) - 1)
    return str(label).strip().upper()


def _record_to_instance(record: dict, release: str, domain: str, task_type: str) -> Instance:
    if task_type == "p":
        input_text = (
            "Given the context, select the text that is the next sequence.\n\n" f"Context:\n{record['context']}"
        )
        correct_letter = str(record["label"]).strip().upper()
    else:
        input_text = (
            "Select the option that correctly completes the sequencing or cloze task.\n\n" f"{record['shuffled_text']}"
        )
        correct_letter = _selection_to_letter(record["label"])

    references: List[Reference] = []
    for letter in ["A", "B", "C", "D"]:
        references.append(
            Reference(
                output=Output(text=str(record[letter]).strip()),
                tags=[CORRECT_TAG] if letter == correct_letter else [],
            )
        )

    return Instance(
        input=Input(text=input_text),
        references=references,
        split=TEST_SPLIT,
        extra_data={
            "release": release,
            "domain": domain,
            "task_type": task_type,
            "task_type_name": TASK_TYPE_NAMES[task_type],
            "source_label": record["label"],
        },
    )


class ArxivRollBenchScenario(Scenario):
    """
    ArxivRollBench is a rolling arXiv benchmark for evaluating recent scientific
    text reasoning. It covers sequencing, cloze, and next-sequence prediction
    tasks across arXiv domains and releases.

    Paper: https://ojs.aaai.org/index.php/AAAI/article/view/41098
    Website: https://arxivrollbench.github.io/
    """

    name = "arxivrollbench"
    description = "A rolling benchmark for recent scientific text reasoning from arXiv papers"
    tags = ["reasoning", "science", "multiple_choice"]

    def __init__(
        self,
        release: str = "all",
        domain: str = "all",
        task_type: str = "all",
        split: Literal["compact", "full"] = "compact",
    ):
        super().__init__()
        if release != "all" and release not in RELEASES:
            raise ValueError(f"Unknown release: {release}")
        domain = DOMAIN_ALIASES.get(domain, domain)
        valid_domains = {domain_name for domain_name, _ in DOMAINS}
        if domain != "all" and domain not in valid_domains:
            raise ValueError(f"Unknown domain: {domain}")
        if task_type != "all" and task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task_type: {task_type}")
        if split not in {"compact", "full"}:
            raise ValueError(f"Unknown split: {split}")

        self.release = release
        self.domain = domain
        self.task_type = task_type
        self.split = split

    def _iter_subsets(self) -> List[Tuple[str, str, str, str]]:
        releases = RELEASES if self.release == "all" else [self.release]
        domains = DOMAINS if self.domain == "all" else [(self.domain, self.domain.replace("_", "-"))]
        task_types = TASK_TYPES if self.task_type == "all" else [self.task_type]
        return [
            (release, domain, hf_domain, task_type)
            for release in releases
            for domain, hf_domain in domains
            for task_type in task_types
        ]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        instances: List[Instance] = []
        for release, domain, hf_domain, task_type in self._iter_subsets():
            dataset = datasets.load_dataset(
                _dataset_path(release, hf_domain, task_type, self.split),
                split="train",
                cache_dir=cache_dir,
            )
            assert isinstance(dataset, datasets.Dataset)
            for record in dataset:
                instances.append(_record_to_instance(record, release, domain, task_type))
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        task_type_display: Optional[str] = None
        if self.task_type != "all":
            task_type_display = TASK_TYPE_NAMES[self.task_type]
        return ScenarioMetadata(
            name=self.name,
            display_name="ArxivRollBench",
            short_display_name="ArxivRollBench",
            description=(
                "ArxivRollBench is a rolling benchmark for evaluating recent scientific "
                "text reasoning over arXiv papers. It covers sequencing, cloze, and "
                "next-sequence prediction tasks across arXiv domains and releases "
                "[(AAAI 2026 paper)](https://ojs.aaai.org/index.php/AAAI/article/view/41098)."
            ),
            taxonomy=TaxonomyInfo(
                task=task_type_display or "multiple-choice scientific text reasoning",
                what="recent scientific text from arXiv papers",
                when="rolling releases from 2024b, 2025a, and 2026a",
                who="arXiv papers across computer science, math, physics, statistics, and related domains",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
