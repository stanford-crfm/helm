import sys
import csv
from typing import Dict, List

from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
)
from helm.common.general import check_file_exists

csv.field_size_limit(sys.maxsize)


class SHCENTMedScenario(Scenario):
    """
    This benchmark dataset was built to assess the capabilities "
    "of an LLM for referral to the Ear, Nose and Throat department.
    """

    name = "shc_ent_med"
    description = (
        "A dataset designed to evaluate performance in "
        "identifying appropriate patient referrals to Ear, Nose, and Throat specialists."
    )
    tags = ["knowledge", "reasoning", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B", "C"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def create_benchmark(self, csv_path) -> Dict[str, str]:
        data = {}
        counter = 1
        with open(csv_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["label"] != "":  # skip rows with character/encoding issues - 79
                    question = row["prompt"]
                    context = row["context"]
                    answer = row["label"]
                    prompt = (
                        f"{counter} Provide an answer to the following question: {question} with the following context:"
                        f" {context} , Answer the question with either 'A' for yes, 'B' for no, or 'C' for no mention."
                        " Do not provide any additional details or response, just a simple A, B, or C response."
                    )
                    data[prompt] = answer
                    counter = counter + 1
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.data_path, msg=f"[SHCENTMedScenario] Required data file not found: '{self.data_path}'")
        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(self.data_path)

        for prompt, answer in benchmark_data.items():
            assert answer in SHCENTMedScenario.POSSIBLE_ANSWER_CHOICES
            references: List[Reference] = [
                Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == answer else [])
                for pred_answer in SHCENTMedScenario.POSSIBLE_ANSWER_CHOICES
            ]
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
