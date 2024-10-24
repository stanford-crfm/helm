import csv
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output

import datasets
from typing import List, Dict
from pathlib import Path
import json
import random 


from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists

class MMLUPROScenario(Scenario):
    """
    This is sample code for the MMLU-PRO Dataset.
    """

    name = "mmlu_pro"
    description = "Massive Multitask Language Understanding PRO"
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject

    def process_csv(self, csv_path: str, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Reading {csv_path}")
        with open(csv_path) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                # Example: ["What color is the sky?", "red", "blue", "green", "B"]
                question, answers, correct_choice = row[0], row[1:-1], row[-1]
                answers_dict = dict(zip(["A", "B", "C", "D", "E"], answers))
                correct_answer: str = answers_dict[correct_choice]

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

                instance = Instance(
                    input=Input(text=question),
                    references=list(map(answer_to_reference, answers)),
                    split=split,
                )
                instances.append(instance)
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        fields, _, label_key, _ = self.load_prompt_construction_settings(output_path)
        cache_dir = str(Path(output_path) / "data")

        train_dataset = datasets.load_dataset(
            "TIGER-Lab/MMLU-Pro", self.subset, trust_remote_code=True, cache_dir=cache_dir, split="train"
        )
        test_dataset = datasets.load_dataset(
            "TIGER-Lab/MMLU-Pro", self.subset, trust_remote_code=True, cache_dir=cache_dir, split="test"
        )
        assert isinstance(train_dataset, datasets.Dataset)
        assert isinstance(test_dataset, datasets.Dataset)

        dataset_splits: Dict[str, datasets.Dataset] = {
            TRAIN_SPLIT: train_dataset,
            TEST_SPLIT: test_dataset,
        }

        # Read all instances
        random.seed(self.random_seed)
        instances: List[Instance] = []
        for split, subset in dataset_splits.items():
            for x in subset:
                assert fields is not None, "Field ordering not loaded"
                prompt: str = "\n".join([f"{field[0]}: {x[field[1]]}" for field in fields])
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=x[label_key]), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances