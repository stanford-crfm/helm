import sys
import csv
from typing import Dict, List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
    ScenarioMetadata,
)
from helm.common.general import check_file_exists

csv.field_size_limit(sys.maxsize)


class SHCCDIMedScenario(Scenario):
    """
    This benchmark dataset was built from Clinical Document Integrity (CDI)
    notes were there are verifications of clinical activities. The idea behind
    it was to assess an LLM capability to answer these questions from previous notes.
    """

    name = "shc_cdi_med"
    description = (
        "CDI-QA is a benchmark constructed from Clinical Documentation Integrity (CDI)"
        "notes. It is used to evaluate a model's ability to verify clinical conditions based on"
        "documented evidence in patient records."
    )
    tags = ["knowledge", "reasoning", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def create_benchmark(self, csv_path) -> Dict[str, str]:
        data = {}
        with open(csv_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                question = row["prompt"]
                context = row["context"]
                answer = row["label"]
                prompt = (
                    f"Provide an answer to the following question: {question} with the following context: {context} , "
                    "Answer the question with either 'A' for yes or 'B' for no. Do not provide any "
                    "additional details or response, just a simple A or B response."
                )
                data[prompt] = answer
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.data_path, msg=f"[SHCCDIMedScenario] Required data file not found: '{self.data_path}'")
        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(self.data_path)

        for prompt, answer in benchmark_data.items():
            assert answer in SHCCDIMedScenario.POSSIBLE_ANSWER_CHOICES
            references: List[Reference] = [
                Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == answer else [])
                for pred_answer in SHCCDIMedScenario.POSSIBLE_ANSWER_CHOICES
            ]
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name="shc_cdi_med",
            display_name="CDI-QA",
            description="CDI-QA is a benchmark constructed from Clinical Documentation Integrity (CDI) "
            "notes. It is used to evaluate a model's ability to verify clinical conditions "
            "based on documented evidence in patient records.",
            taxonomy=TaxonomyInfo(
                task="Classification",
                what="Answer verification questions from CDI notes",
                when="Any",
                who="Hospital Admistrator",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
