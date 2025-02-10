import sys
import csv
import os
from typing import Dict, List

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
)

csv.field_size_limit(sys.maxsize)

class SHCENTMedScenario(Scenario):
    """
    Benchmark derived from ENT
    """
    name = "shc_ent_med"
    description = "A benchmark for evaluating LLM patient referral capabilities"
    tags = ["knowledge", "reasoning", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = ["A", "B", "C"]
    def create_benchmark(self, csv_path)->Dict[str, str]:
        data = {}
        counter = 1
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['label'] != '':   #skip rows with character/encoding issues - 79
                    question = row['prompt']
                    context = row['context']
                    answer = row['label']
                    prompt = f"{counter} Provide an answer to the following {question} with the following context: {context} , Answer the question with either 'A' for yes, 'B' for no, or 'C' for no mention. Do not provide any additional details or response, just a simple A, B, or C response."
                    data[prompt] = answer
                    counter=counter+1
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = "/dbfs/mnt/azure_adbfs/Files/medhelm/medhelm-ENT-dataset_filtered.csv"

        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(data_path)

        for prompt, answer in benchmark_data.items():
            assert answer in SHCENTMedScenario.POSSIBLE_ANSWER_CHOICES
            references: List[Reference] = [
                        Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == answer else [])
                        for pred_answer in SHCENTMedScenario.POSSIBLE_ANSWER_CHOICES
                    ]
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references, #[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )
        
        return instances