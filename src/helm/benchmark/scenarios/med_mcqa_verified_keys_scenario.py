"""Verified-keys track for MedMCQA, as a HELM scenario.

Modeled on the upstream MedMCQA scenario
(src/helm/benchmark/scenarios/med_mcqa_scenario.py in
stanford-crfm/helm). The difference is the answer key: every key in
this track was re-derived blind, from first principles, and sealed
before the official key was seen. The official key is carried as
labelled metadata (RAW), never as the grading target.

Each instance carries its full integrity record as extra_data —
disposition, defect flags, integrity class, raw key letter,
verified key letter, and the scorable flag — so a verified-only
metric can filter and audit item by item. The verified answer
carries CORRECT_TAG only where the item is scorable on the
verified track.

Data: the public slice file on Hugging Face,
agent-embassy/medmcqa-checked-answers-slice-1
(verified-keys-slice-v2.jsonl, 6,800 questions, first slice of a
running census of all 182,822 MedMCQA training questions;
175,878 remain).

Research material only. Not clinical guidance. Not validated for
patient care.
"""
import json
import os
from typing import Dict, List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Output,
    ScenarioMetadata,
)


class MedMCQAVerifiedKeysScenario(Scenario):
    """MedMCQA with blind-verified answer keys (first slice)."""

    DATASET_DOWNLOAD_URL: str = (
        "https://huggingface.co/datasets/agent-embassy/"
        "medmcqa-checked-answers-slice-1/resolve/main/verified-keys-slice-v2.jsonl"
    )
    LETTERS: List[str] = ["A", "B", "C", "D"]

    name = "med_mcqa_verified_keys"
    description = (
        "MedMCQA is a multiple-choice question answering (MCQA) dataset designed to address "
        "real-world medical entrance exam questions. This track re-derives every answer key "
        "blind, from first principles, sealed before the official key was seen; the official "
        "key is carried as labelled metadata and the verified key is the grading target "
        "where the item is scorable."
    )
    tags = ["question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=False,
        )

        instances: List[Instance] = []
        with open(data_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row: Dict = json.loads(line.rstrip())

                verified_letter = (row.get("verified_key") or {}).get("letter")
                scorable = bool(row.get("scorable_on_verified_track"))
                references: List[Reference] = [
                    Reference(
                        Output(text=row["options"][letter]),
                        tags=[CORRECT_TAG] if (scorable and letter == verified_letter) else [],
                    )
                    for letter in self.LETTERS
                ]
                instance: Instance = Instance(
                    input=Input(text=row["stem"]),
                    references=references,
                    split=TEST_SPLIT,
                    id=row["source_id"],
                    extra_data={
                        "package": row["package"],
                        "item": row["item"],
                        "source_index": row["source_index"],
                        "source_split": row["split"],
                        "raw_key_letter": (row.get("raw_key") or {}).get("letter"),
                        "verified_key_letter": verified_letter,
                        "verified_key_status": row.get("verified_key_status"),
                        "disposition": row.get("disposition"),
                        "integrity_class": row.get("integrity_class"),
                        "flags": row.get("flags", []),
                        "scorable_on_verified_track": scorable,
                    },
                )
                instances.append(instance)

        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name="med_mcqa_verified_keys",
            display_name="MedMCQA (Verified Keys)",
            description="MedMCQA is a multiple-choice question answering (MCQA) dataset designed to "
            "address real-world medical entrance exam questions. This track grades against "
            "blind-verified answer keys: each key was re-derived from first principles and "
            "sealed before the official key was seen, and items that cannot be scored as "
            "clean knowledge (disputed, defective, or missing their true answer) are "
            "labelled and carry no grading target.",
            taxonomy=TaxonomyInfo(task="question answering", what="n/a", when="n/a", who="n/a", language="English"),
            main_metric="exact_match",
            main_split="test",
        )
