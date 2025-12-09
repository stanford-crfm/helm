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


class SHCPROXYMedScenario(Scenario):
    """
    This dataset features messages sent by proxy users and non proxy users, for evaluation of
    LLM capabilities to determine the sender. From publication: https://doi.org/10.1001/jamapediatrics.2024.4438
    """

    name = "shc_proxy_med"
    description = (
        "ProxySender is a benchmark composed of patient portal messages received by clinicians."
        "It evaluates whether the message was sent by the patient or by a proxy user (e.g., parent,"
        "spouse), which is critical for understanding who is communicating with healthcare"
        "providers."
    )
    tags = ["knowledge", "reasoning", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B", "C"]

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
                    "You are reviewing a clinical messages in order to determine if they have been "
                    f"sent by a proxy user. Please determine the following: {question} with the "
                    f"following context: {context} , Answer the question with a 'A' for yes or 'B' "
                    "for no. Do not provide any additional details or response, just a simple A or B response."
                )
                data[prompt] = answer
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.data_path, msg=f"[SHCPROXYMedScenario] Required data file not found: '{self.data_path}'")
        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(self.data_path)

        for prompt, answer in benchmark_data.items():
            assert answer in SHCPROXYMedScenario.POSSIBLE_ANSWER_CHOICES
            references: List[Reference] = [
                Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == answer else [])
                for pred_answer in SHCPROXYMedScenario.POSSIBLE_ANSWER_CHOICES
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
            name="shc_proxy_med",
            display_name="ProxySender",
            description="ProxySender is a benchmark composed of patient portal messages received by "
            "clinicians. It evaluates whether the message was sent by the patient or by a "
            "proxy user (e.g., parent, spouse), which is critical for understanding who is "
            "communicating with healthcare providers. [(Tse G, et al., "
            "2025)](https://doi.org/10.1001/jamapediatrics.2024.4438).",
            taxonomy=TaxonomyInfo(
                task="Classification",
                what="Classify if a document was sent by a proxy user",
                when="Any",
                who="Clinician, Caregiver",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
