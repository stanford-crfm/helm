import csv
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class MMLUScenario(Scenario):
    """
    The Massive Multitask Language Understanding benchmark from this paper:

    - https://arxiv.org/pdf/2009.03300.pdf

    Code is adapted from:

    - https://github.com/hendrycks/test/blob/master/evaluate.py
    - https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py

    We prompt models using the following format

        <input>                  # train
        A. <reference>
        B. <reference>
        C. <reference>
        D. <reference>
        Answer: <A/B/C/D>

        x N (N-shot)

        <input>                  # test
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer:

    For example (from mmlu:anatomy), we have:

        The pleura
        A. have no sensory innervation.
        B. are separated by a 2 mm space.
        C. extend into the neck.
        D. are composed of respiratory epithelium.
        Answer: C

        Which of the following terms describes the body's ability to maintain its normal state?
        A. Anabolism
        B. Catabolism
        C. Tolerance
        D. Homeostasis
        Answer:

    Target: D
    """

    name = "mmlu"
    description = "Massive Multitask Language Understanding"
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject

    def download_mmlu(self, path: str):
        ensure_file_downloaded(
            source_url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
            target_path=path,
            unpack=True,
            unpack_type="untar",
        )

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
        # Download the raw data
        data_path: str = os.path.join(output_path, "data")
        self.download_mmlu(data_path)

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "auxiliary_train": TRAIN_SPLIT,
            "dev": TRAIN_SPLIT,
            "val": VALID_SPLIT,
            "test": TEST_SPLIT,
        }
        for split in splits:
            csv_path: str = os.path.join(data_path, split, f"{self.subject}_{split}.csv")
            if not os.path.exists(csv_path):
                hlog(f"{csv_path} doesn't exist, skipping")
                continue
            instances.extend(self.process_csv(csv_path, splits[split]))

        return instances
