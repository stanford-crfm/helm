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

csv.field_size_limit(sys.maxsize)


class SHCCDIMedScenario(Scenario):
    """
    This benchmark dataset was built from Clinical Document Integrity (CDI)
    notes were there are verifications of clinical activities. The idea behind
    it was to assess an LLM capability to answer these questions from previous notes.
    """

    name = "shc_cdi_med"
    description = (
        "A dataset built from Clinical Document Integrity (CDI) notes, to assess "
        "the ability to answer verification questions from previous notes."
    )
    tags = ["knowledge", "reasoning", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B"]

    def create_benchmark(self, csv_path) -> Dict[str, str]:
        data = {}
        with open(csv_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                question = row["prompt"]
                context = row["context"]
                answer = row["label"]
                prompt = (
                    f"Provide an answer to the following {question} with the following context: {context} , "
                    "Answer the question with either 'A' for yes or 'B' for no. Do not provide any "
                    "additional details or response, just a simple A or B response."
                )
                data[prompt] = answer
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = "/dbfs/mnt/azure_adbfs/Files/medhelm/medhelm-CDI-dataset_filtered.csv"

        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(data_path)

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
