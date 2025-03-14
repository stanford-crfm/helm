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


class SHCPTBMMedScenario(Scenario):
    """
    This dataset contains clinical notes from primary care visit encounters of
    children ages 4-6 years old with ADHD seen at Stanford's community-based primary
    care network, Packard Children's Health Alliance, between 2015-2019. In this classification
    task, the LLM is tasked with classifying whether the note contains clinician recommendation
    for parent training in behavior management, which is the first-line evidence-based treatment
    for young children with ADHD. From publication: https://doi.org/10.1093/jamia/ocae001
    """

    name = "shc_ptbm_med"
    description = (
        "A dataset that classifies whether a clinical note contains a clinician "
        "recommendation for parent training in behavior management, which is the first-line "
        "evidence-based treatment for young children with ADHD."
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
                    "You are reviewing a clinical note from health records of children with "
                    "attention deficit hyperactivity disorder (ADHD) and classifying mentions of "
                    f"behavioral therapy. Provide an answer to the following {question} with the "
                    f"following context: {context} , Answer the question with a 'A' for yes or 'B' "
                    "for no. Do not provide any additional details or response, just a simple A or B response."
                )
                data[prompt] = answer
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = "/dbfs/mnt/azure_adbfs/Files/medhelm/medhelm-PTBM-dataset_filtered.csv"

        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(data_path)

        for prompt, answer in benchmark_data.items():
            assert answer in SHCPTBMMedScenario.POSSIBLE_ANSWER_CHOICES
            references: List[Reference] = [
                Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == answer else [])
                for pred_answer in SHCPTBMMedScenario.POSSIBLE_ANSWER_CHOICES
            ]
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,  # [Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
