import json
import os
from typing import List
from common.general import ensure_file_downloaded, ensure_directory_exists
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG
from functools import partial

CLM_CORRECT_TAG = "clm_correct"


class CommonSenseQAScenario(Scenario):
    """
    Unified interface for all CommonSense QA scenarios.

    The "HellaSwag" benchmark from this paper:
        https://arxiv.org/pdf/1905.07830.pdf

    The "OpenBookQA" benchmark from this paper:
        https://aclanthology.org/D18-1260.pdf
    """

    name = "commonsense_qa"
    description = "Unified interface for all CommonSense QA scenarios."
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, dataset, method):
        self.dataset = dataset
        assert self.dataset in ["hellaswag", "openbookqa"]
        self.method = method
        assert self.method in ["mcqa", "clm"]

    @staticmethod
    def process_hellaswag_item(item):
        ctx_b_fixed = item["ctx_b"][0].upper() + item["ctx_b"][1:] if len(item["ctx_b"]) != 0 else ""

        question = f"{item['activity_label']}: {item['ctx_a']} {ctx_b_fixed}"
        answers = item["endings"]
        correct_answer = answers[item["label"]]

        assert len(answers) == 4
        return question, answers, correct_answer

    @staticmethod
    def process_openbookqa_item(item):
        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}

        question = item["question"]["stem"]
        answers = [answer["text"] for answer in item["question"]["choices"]]
        correct_choice = letter2idx[item["answerKey"]]
        correct_answer = answers[correct_choice]

        assert len(answers) == 4
        assert item["question"]["choices"][correct_choice]["label"] == item["answerKey"]
        return question, answers, correct_answer

    def download_dataset(self):
        # Download the raw data
        data_path = os.path.join(self.output_path, "data", self.dataset)
        ensure_directory_exists(data_path)

        if self.dataset == "hellaswag":
            url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_{}.jsonl"
            for split in ["train", "val", "test"]:
                ensure_file_downloaded(
                    source_url=url.format(split), target_path=os.path.join(data_path, f"hellaswag_{split}.jsonl"),
                )
        elif self.dataset == "openbookqa":
            ensure_file_downloaded(
                source_url="https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip",
                target_path=os.path.join(data_path, "OpenBookQA-V1-Sep2018"),
                unpack=True,
                unpack_type="unzip",
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def load_dataset(self) -> List[List[str]]:
        data_path = os.path.join(self.output_path, "data", self.dataset)

        if self.dataset == "hellaswag":
            split2file = {
                split: os.path.join(data_path, f"hellaswag_{split}.jsonl") for split in ["train", "val"]
            }  # Ignore HellaSwag test set because no label information
            item_process_func = self.process_hellaswag_item
        elif self.dataset == "openbookqa":
            split2file = {
                split: os.path.join(data_path, "OpenBookQA-V1-Sep2018", "Data", "Main", f"{split}.jsonl")
                for split in ["train", "val", "test"]
            }
            item_process_func = self.process_openbookqa_item
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        data = []
        for split in split2file:
            file_path = split2file[split]
            if not os.path.exists(file_path):
                hlog(f"{file_path} doesn't exist, skipping")
                continue

            hlog(f"Reading {file_path}")
            with open(file_path) as f:
                for line in f:
                    item = json.loads(line)
                    question, answers, correct_answer = item_process_func(item)
                    data.append([question, answers, correct_answer, split])
        return data

    def get_instances(self) -> List[Instance]:
        self.download_dataset()
        data = self.load_dataset()

        splits = {
            "train": TRAIN_TAG,
            "val": VALID_TAG,
            "test": TEST_TAG,
        }

        instances = []

        def answer_to_reference(answer, correct_tag):
            return Reference(output=answer, tags=[correct_tag] if answer == correct_answer else [])

        for (question, answers, correct_answer, split) in data:
            if self.method == "mcqa":
                answer_to_reference_mcqa = partial(answer_to_reference, correct_tag=CORRECT_TAG)
                instance = Instance(
                    input=question, references=list(map(answer_to_reference_mcqa, answers)), tags=[splits[split]],
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
                        input=f"Answer: {answer}", references=[answer_to_reference_clm(answer)], tags=[splits[split]],
                    )
                    instances += [instance1, instance2]
        return instances
