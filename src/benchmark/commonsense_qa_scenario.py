import json
import os
from typing import List
from common.general import ensure_file_downloaded, ensure_directory_exists
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


class HellaSwagScenario(Scenario):
    """
    The "HellaSwag: Can a Machine Really Finish Your Sentence?" benchmark from this paper:

        https://arxiv.org/pdf/1905.07830.pdf

    Code is adapted from:

        https://github.com/hendrycks/test/blob/master/evaluate.py
        https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py
    """

    name = "hellaswag"
    description = "HellaSwag: Can a Machine Really Finish Your Sentence?"
    tags = ["knowledge", "multiple_choice"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        # Read all the instances
        instances = []
        splits = {
            "train": TRAIN_TAG,
            "val": VALID_TAG,
            # "test": TEST_TAG # <Yuhui>: hellaswag test set does not have label information
        }
        for split in splits:
            ensure_file_downloaded(
                source_url=f"https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_{split}.jsonl",
                target_path=os.path.join(data_path, f"hellaswag_{split}.jsonl"),
            )

            jsonl_path = os.path.join(data_path, f"hellaswag_{split}.jsonl")
            if not os.path.exists(jsonl_path):
                hlog(f"{jsonl_path} doesn't exist, skipping")
                continue

            hlog(f"Reading {jsonl_path}")
            with open(jsonl_path) as f:
                for line in f:
                    """
                    Example:
                    {
                        "ind": 24,
                        "activity_label": "Roof shingle removal",
                        "ctx_a": "A man is sitting on a roof.",
                        "ctx_b": "he",
                        "ctx": "A man is sitting on a roof. he",
                        "split": "val",
                        "split_type": "indomain",
                        "label": 3,
                        "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.",
                                    "is holding a rubik's cube.", "starts pulling up roofing on a roof."],
                        "source_id": "activitynet~v_-JhWjGDPHMY"
                    }
                    """
                    item = json.loads(line)
                    assert item["split"] == split
                    question, answers, correct_choice = item["ctx"], item["endings"], item["label"]
                    correct_answer = answers[correct_choice]

                    def answer_to_reference(answer):
                        return Reference(output=answer, tags=[CORRECT_TAG] if answer == correct_answer else [])

                    instance = Instance(
                        input=question, references=list(map(answer_to_reference, answers)), tags=[splits[split]],
                    )
                    instances.append(instance)

        return instances


class OpenBookQAScenario(Scenario):
    """
    The "OpenBookQA: Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering"
    benchmark from this paper:

        https://aclanthology.org/D18-1260.pdf

    Code is adapted from:

        https://github.com/hendrycks/test/blob/master/evaluate.py
        https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py
    """

    name = "openbookqa"
    description = "OpenBookQA: Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering"
    tags = ["knowledge", "multiple_choice"]

    def __init__(self):
        pass

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_file_downloaded(
            source_url="https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip",
            target_path=data_path,
            unpack=True,
            unpack_type="unzip",
        )

        # Read all the instances
        instances = []
        splits = {
            "train": TRAIN_TAG,
            "val": VALID_TAG,
            "test": TEST_TAG,
        }
        for split in splits:
            jsonl_path = os.path.join(data_path, "Data", "Main", f"{split}.jsonl")
            if not os.path.exists(jsonl_path):
                hlog(f"{jsonl_path} doesn't exist, skipping")
                continue

            hlog(f"Reading {jsonl_path}")
            with open(jsonl_path) as f:
                for line in f:
                    """
                    Example:
                    {
                        "id": "8-376",
                        "question": {
                            "stem": "Frilled sharks and angler fish live far beneath the surface
                                     of the ocean, which is why they are known as",
                            "choices": [{"text": "Deep sea animals", "label": "A"},
                                        {"text": "fish", "label": "B"},
                                        {"text": "Long Sea Fish", "label": "C"},
                                        {"text": "Far Sea Animals", "label": "D"}]
                        },
                        "answerKey": "A"
                    }
                    """
                    item = json.loads(line)
                    question, answers, correct_choice = (
                        item["question"]["stem"],
                        item["question"]["choices"],
                        item["answerKey"],
                    )

                    def answer_to_reference(answer):
                        return Reference(
                            output=answer["text"], tags=[CORRECT_TAG] if answer["label"] == correct_choice else []
                        )

                    instance = Instance(
                        input=question, references=list(map(answer_to_reference, answers)), tags=[splits[split]],
                    )
                    instances.append(instance)

        return instances
