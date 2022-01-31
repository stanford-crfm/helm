import json
import os
from typing import List
from common.general import ensure_file_downloaded, ensure_directory_exists
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG
from functools import partial

CLM_CORRECT_TAG = "clm_correct"


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

    def __init__(self, method):
        self.method = method
        assert self.method in ["mcqa", "clm"]

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        # Read all the instances in the train and val set.
        # Ignore test set because HellaSwag test set does not have label information.
        instances = []
        splits = {
            "train": TRAIN_TAG,
            "val": VALID_TAG,
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
                    ctx_b_fixed = item["ctx_b"][0].upper() + item["ctx_b"][1:] if len(item["ctx_b"]) != 0 else ""
                    question, answers, correct_choice = (
                        f"{item['activity_label']}: {item['ctx_a']} {ctx_b_fixed}",
                        item["endings"],
                        item["label"],
                    )
                    assert len(answers) == 4
                    correct_answer = answers[correct_choice]

                    def answer_to_reference(answer, correct_tag):
                        return Reference(output=answer, tags=[correct_tag] if answer == correct_answer else [])

                    if self.method == "mcqa":
                        answer_to_reference_mcqa = partial(answer_to_reference, correct_tag=CORRECT_TAG)
                        instance = Instance(
                            input=question,
                            references=list(map(answer_to_reference_mcqa, answers)),
                            tags=[splits[split]],
                        )
                        instances.append(instance)
                    elif self.method == "clm":
                        answer_to_reference_clm = partial(answer_to_reference, correct_tag=CLM_CORRECT_TAG)
                        for answer in answers:
                            instance1 = Instance(
                                input=f"{question} {answer}",
                                references=[answer_to_reference_clm(answer)],
                                tags=[splits[split]],
                            )
                            instance2 = Instance(
                                input=f"Answer: {answer}",
                                references=[answer_to_reference_clm(answer)],
                                tags=[splits[split]],
                            )
                            instances += [instance1, instance2]

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
                    assert len(answers) == 4

                    def answer_to_reference(answer):
                        return Reference(
                            output=answer["text"], tags=[CORRECT_TAG] if answer["label"] == correct_choice else []
                        )

                    instance = Instance(
                        input=question, references=list(map(answer_to_reference, answers)), tags=[splits[split]],
                    )
                    instances.append(instance)

        return instances
