import csv
import os
from typing import Dict, List

from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG
from .mmlu_scenario import MMLUScenario


class InteractiveQAMMLUScenario(MMLUScenario):
    """
    The Massive Multitask Language Understanding benchmark from this paper:

        https://arxiv.org/pdf/2009.03300.pdf

    For InteractiveQA, we used a small subset of the original test set.
    """

    name = "interactive_qa_mmlu"
    description = "Massive Multitask Language Understanding (InteractiveQA subset)"
    tags = ["knowledge", "multiple_choice"]

    VALID_SUBJECTS: List[str] = [
        "college_chemistry",
        "global_facts",
        "miscellaneous",
        "nutrition",
        "us_foreign_policy",
    ]
    INTERACTIVE_QA_DIR: str = "interactive_qa"

    def __init__(self, subject: str):
        assert subject in InteractiveQAMMLUScenario.VALID_SUBJECTS, f"Invalid subject for Interactive QA: {subject}"
        super().__init__(subject)

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
            target_path=data_path,
            unpack=True,
            unpack_type="untar",
        )
        ensure_file_downloaded(
            source_url="https://worksheets.codalab.org/rest/bundles/0x4d49146fe16c4559bc64af6cbc04992d/"
            "contents/blob/",
            target_path=os.path.join(data_path, "test", InteractiveQAMMLUScenario.INTERACTIVE_QA_DIR),
            unpack=True,
            unpack_type="untar",
        )

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "auxiliary_train": TRAIN_SPLIT,
            "dev": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }
        for split in splits:
            split_path: str = os.path.join(data_path, split)
            if split == "test":
                split_path = os.path.join(split_path, InteractiveQAMMLUScenario.INTERACTIVE_QA_DIR)

            csv_path: str = os.path.join(split_path, f"{self.subject}_{split}.csv")
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
                        input=question,
                        references=list(map(answer_to_reference, answers)),
                        split=splits[split],
                    )
                    instances.append(instance)

        return instances
