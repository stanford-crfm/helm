"""JFinQA: Japanese Financial Numerical Reasoning QA Benchmark.

Data source:
https://huggingface.co/datasets/ajtgjmdjp/jfinqa

JFinQA is a benchmark for numerical reasoning over Japanese corporate
financial disclosures. It contains 1,000 questions across three subtasks
—numerical reasoning (550), consistency checking (200), and temporal
reasoning (250)—drawn from 68 companies' EDINET filings covering
J-GAAP, IFRS, and US-GAAP.
"""

import os
from typing import Any, Dict, List

from datasets import load_dataset

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


class JFinQAScenario(Scenario):
    """Japanese Financial Numerical Reasoning QA."""

    name = "jfinqa"
    description = (
        "JFinQA: Japanese Financial Numerical Reasoning QA — "
        "1,000 questions across numerical reasoning, consistency checking, "
        "and temporal reasoning from 68 companies' EDINET filings."
    )
    tags = ["question_answering", "finance", "japanese"]

    HF_DATASET_ID = "ajtgjmdjp/jfinqa"
    SUBSETS = ("numerical_reasoning", "consistency_checking", "temporal_reasoning")

    @staticmethod
    def _format_table(headers: List[str], rows: List[List[str]]) -> str:
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        return "\n".join([header_line, sep_line, *row_lines])

    @staticmethod
    def _build_input(row: Dict[str, Any]) -> str:
        parts: List[str] = []

        pre_text = row.get("pre_text", [])
        if pre_text:
            parts.append("\n".join(pre_text))

        headers = row.get("table_headers", [])
        rows = row.get("table_rows", [])
        if headers:
            parts.append(JFinQAScenario._format_table(headers, rows))

        post_text = row.get("post_text", [])
        if post_text:
            parts.append("\n".join(post_text))

        question = row.get("question", "")
        parts.append(f"Question: {question}")

        return "\n\n".join(parts)

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        instances: List[Instance] = []
        for subset in self.SUBSETS:
            dataset = load_dataset(
                self.HF_DATASET_ID,
                subset,
                split="test",
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            for row in dataset:
                input_text = self._build_input(row)
                answer = str(row["answer"])
                instances.append(
                    Instance(
                        input=Input(text=input_text),
                        references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                        split=TEST_SPLIT,
                        id=str(row.get("id", "")),
                    )
                )
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="jfinqa",
            display_name="JFinQA",
            short_display_name="JFinQA",
            description=self.description,
            taxonomy=TaxonomyInfo(
                task="question answering with numeric reasoning",
                what="Japanese corporate financial reports (EDINET/XBRL)",
                when="2023 to 2024",
                who="financial experts",
                language="Japanese",
            ),
            main_metric="float_equiv",
            main_split="test",
        )
