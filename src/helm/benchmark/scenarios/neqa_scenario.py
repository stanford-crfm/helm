import json
import os
from typing import List

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
)


class NeQAScenario(Scenario):
    """
    Interface for the NeQA (Negated QA) scenario.
      [Link] # TODO: add link to the work
    """

    name = "neqa"
    description = "Interface for the NeQA scenario."
    tags = ["negation", "knowledge", "multiple_choice"]

    def __init__(self):
        super().__init__()


    @staticmethod
    def process_item(item):
        letter2idx = {"A": 0, "B": 1}

        question = item["question"]["stem"]
        answers = [answer["text"] for answer in item["question"]["choices"]]
        correct_choice = letter2idx[item["answerKey"]]
        correct_answer = answers[correct_choice]

        assert len(answers) == 2
        assert item["question"]["choices"][correct_choice]["label"] == item["answerKey"]
        return question, answers, correct_answer


    def download_dataset(self):
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)
        url = "https://raw.githubusercontent.com/yuhui-zh15/NeQA/main/final_submission/cache/final_submission.jsonl"  # TODO: update the link in the future
        ensure_file_downloaded(source_url=url, target_path=os.path.join(data_path, f"neqa.jsonl"))

    def load_dataset(self) -> List[List[str]]:
        file_path = os.path.join(self.output_path, "data", "neqa.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = []
        split = "test"  # TODO: update the split if we have train / val in the future
        hlog(f"Reading {file_path}")
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                question, answers, correct_answer = self.process_item(item)
                data.append([question, answers, correct_answer, split])
        return data

    def get_instances(self) -> List[Instance]:
        self.download_dataset()
        data = self.load_dataset()

        splits = {
            "train": TRAIN_SPLIT,
            "val": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        instances: List[Instance] = []

        def answer_to_reference(answer):
            return Reference(output=answer, tags=[CORRECT_TAG] if answer == correct_answer else [])

        for question_id, (question, answers, correct_answer, split) in enumerate(data):
            instance = Instance(
                input=question,
                references=list(map(answer_to_reference, answers)),
                split=splits[split],
            )
            instances.append(instance)
        return instances
