import csv
import os
from typing import List
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


class MMLUScenario(Scenario):
    """
    The Massive Multitask Language Understanding benchmark from this paper:

        https://arxiv.org/pdf/2009.03300.pdf

    Code is adapted from:

        https://github.com/hendrycks/test/blob/master/evaluate.py
        https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py
    """

    name = "mmlu"
    description = "Massive Multitask Language Understanding"
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, subject: str):
        self.subject = subject

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
            target_path=data_path,
            unpack=True,
            unpack_type="untar",
        )

        # Read all the instances
        instances = []
        splits = {
            "auxiliary_train": TRAIN_TAG,
            "dev": TRAIN_TAG,
            "val": VALID_TAG,
            "test": TEST_TAG,
        }
        for split in splits:
            csv_path = os.path.join(data_path, split, f"{self.subject}_{split}.csv")
            if not os.path.exists(csv_path):
                hlog(f"{csv_path} doesn't exist, skipping")
                continue

            hlog(f"Reading {csv_path}")
            with open(csv_path) as f:
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    # Example: ["What color is the sky?", "red", "blue", "green", "B"]
                    question, answers, correct_choice = row[0], row[1:-1], row[-1]
                    answers_dict = dict(zip(["A", "B", "C", "D", "E"], answers))
                    correct_answer = answers_dict[correct_choice]

                    def answer_to_reference(answer):
                        return Reference(output=answer, tags=[CORRECT_TAG] if answer == correct_answer else [])

                    instance = Instance(
                        input=question, references=list(map(answer_to_reference, answers)), tags=[splits[split]],
                    )
                    instances.append(instance)

        return instances
