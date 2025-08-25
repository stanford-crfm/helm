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


class SHCBMTMedScenario(Scenario):
    """
    This benchmark dataset was built from a patient status gold-standard
    for specific questions asked after a bone marrow transplant has taken place.
    """

    name = "shc_bmt_med"
    description = (
        "BMT-Status is a benchmark composed of clinical notes and associated binary questions"
        "related to bone marrow transplant (BMT), hematopoietic stem cell transplant (HSCT),"
        "or hematopoietic cell transplant (HCT) status. The goal is to determine whether the"
        "patient received a subsequent transplant based on the provided clinical documentation."
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
                    f"Provide an answer to the following question: {question} with the following context: {context} "
                    ", Answer the question with a 'A' for yes or 'B' for no. Do not provide any additional "
                    "details or response, just a simple A or B response."
                )
                data[prompt] = answer
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.data_path, msg=f"[SHCBMTMedScenario] Required data file not found: '{self.data_path}'")
        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(self.data_path)

        for prompt, answer in benchmark_data.items():
            assert answer in SHCBMTMedScenario.POSSIBLE_ANSWER_CHOICES
            references: List[Reference] = [
                Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == answer else [])
                for pred_answer in SHCBMTMedScenario.POSSIBLE_ANSWER_CHOICES
            ]
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,  # [Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name="shc_bmt_med",
            display_name="BMT-Status",
            description="BMT-Status is a benchmark composed of clinical notes and associated binary "
            "questions related to bone marrow transplant (BMT), hematopoietic stem cell "
            "transplant (HSCT), or hematopoietic cell transplant (HCT) status. The goal is "
            "to determine whether the patient received a subsequent transplant based on the "
            "provided clinical documentation.",
            taxonomy=TaxonomyInfo(
                task="question answering",
                what="Answer bone marrow transplant questions",
                when="Any",
                who="Researcher",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
