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


class SHCSequoiaMedScenario(Scenario):
    """
    Benchmark derived from manually curated answers to several questions for Sequoia clinic referrals
    """

    name = "shc_sequoia_med"
    description = (
        "A dataset containing manually curated answers to questions regarding patient referrals to the Sequoia clinic."
    )
    tags = ["knowledge", "reasoning", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def create_benchmark(self, csv_path) -> Dict[str, str]:
        data = {}
        counter = 1
        with open(csv_path, "r") as file:
            reader = csv.DictReader(file)  # , quoting=csv.QUOTE_MINIMAL
            for row in reader:
                question = row["question"]
                context = row["context"]
                answer = row["label"]
                prompt = (
                    f" {counter} Provide an answer to the following question: {question} with the following context:"
                    f" {context} , Answer the question with a 'A' for yes or 'B' for no. Do not provide any "
                    "additional details or response, just a simple A or B response."
                )
                data[prompt] = answer
                counter += 1
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(
            self.data_path, msg=f"[SHCSequoiaMedScenario] Required data file not found: '{self.data_path}'"
        )
        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(self.data_path)

        for prompt, answer in benchmark_data.items():
            assert answer in SHCSequoiaMedScenario.POSSIBLE_ANSWER_CHOICES
            references: List[Reference] = [
                Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == answer else [])
                for pred_answer in SHCSequoiaMedScenario.POSSIBLE_ANSWER_CHOICES
            ]
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,  # [Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
