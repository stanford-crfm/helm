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
    Input,
    Output,
)


class CommonSenseScenario(Scenario):
    """
    Unified interface for all CommonSense scenarios.

    - The "HellaSwag" benchmark from this paper:
      https://arxiv.org/pdf/1905.07830.pdf

    - The "OpenBookQA" benchmark from this paper:
      https://aclanthology.org/D18-1260.pdf

    - The "CommonSenseQA" benchmark from this paper:
      https://arxiv.org/pdf/1811.00937.pdf

    - The "PIQA" benchmark from this paper:
      https://arxiv.org/pdf/1911.11641.pdf

    - The "SIQA" benchmark from this paper:
      https://arxiv.org/pdf/1904.09728.pdf
    """

    name = "commonsense"
    description = "Unified interface for all CommonSense scenarios."
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        assert self.dataset in ["hellaswag", "openbookqa", "commonsenseqa", "piqa", "siqa"]

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

    @staticmethod
    def process_commonsenseqa_item(item):
        # Note: question concept field is not used: item["question"]["question_concept"]
        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        question = item["question"]["stem"]
        answers = [answer["text"] for answer in item["question"]["choices"]]
        correct_choice = letter2idx[item["answerKey"]]
        correct_answer = answers[correct_choice]

        assert len(answers) == 5
        assert item["question"]["choices"][correct_choice]["label"] == item["answerKey"]
        return question, answers, correct_answer

    @staticmethod
    def process_piqa_item(item):
        question = item["goal"]
        answers = [item["sol1"], item["sol2"]]
        correct_choice = item["label"]
        correct_answer = answers[correct_choice]

        assert len(item) == 4
        assert correct_choice in [0, 1]
        return question, answers, correct_answer

    @staticmethod
    def process_siqa_item(item):
        question = f"{item['context']} {item['question']}"
        answers = [item["answerA"], item["answerB"], item["answerC"]]
        correct_choice = item["label"] - 1
        correct_answer = answers[correct_choice]

        assert len(item) == 6
        assert correct_choice in [0, 1, 2]
        return question, answers, correct_answer

    def download_dataset(self):
        # Download the raw data
        data_path = os.path.join(self.output_path, "data", self.dataset)
        ensure_directory_exists(data_path)

        if self.dataset == "hellaswag":
            url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_{}.jsonl"
            for split in ["train", "val", "test"]:
                ensure_file_downloaded(
                    source_url=url.format(split),
                    target_path=os.path.join(data_path, f"hellaswag_{split}.jsonl"),
                )
        elif self.dataset == "openbookqa":
            ensure_file_downloaded(
                source_url="https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip",
                target_path=os.path.join(data_path, "OpenBookQA-V1-Sep2018"),
                unpack=True,
                unpack_type="unzip",
            )
        elif self.dataset == "commonsenseqa":
            url = "https://s3.amazonaws.com/commensenseqa/{}_rand_split.jsonl"
            split_mapping = {"train": "train", "val": "dev"}
            for split in ["train", "val"]:
                ensure_file_downloaded(
                    source_url=url.format(split_mapping[split]),
                    target_path=os.path.join(data_path, f"commonsenseqa_{split}.jsonl"),
                )
        elif self.dataset == "piqa":
            url = "https://yonatanbisk.com/piqa/data/{}"
            split_mapping = {"train": "train", "val": "valid"}
            for split in ["train", "val"]:
                ensure_file_downloaded(
                    source_url=url.format(f"{split_mapping[split]}.jsonl"),
                    target_path=os.path.join(data_path, f"piqa_{split}_raw.jsonl"),
                )
                ensure_file_downloaded(
                    source_url=url.format(f"{split_mapping[split]}-labels.lst"),
                    target_path=os.path.join(data_path, f"piqa_{split}_labels.lst"),
                )
                data = [json.loads(line) for line in open(os.path.join(data_path, f"piqa_{split}_raw.jsonl"))]
                labels = [int(line.strip()) for line in open(os.path.join(data_path, f"piqa_{split}_labels.lst"))]
                assert len(data) == len(labels)
                for item, label in zip(data, labels):
                    item["label"] = label
                with open(os.path.join(data_path, f"piqa_{split}.jsonl"), "w") as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
        elif self.dataset == "siqa":
            ensure_file_downloaded(
                source_url="https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
                target_path=os.path.join(data_path, "socialiqa-train-dev"),
                unpack=True,
                unpack_type="unzip",
            )
            split_mapping = {"train": "train", "val": "dev"}
            for split in ["train", "val"]:
                data = [
                    json.loads(line)
                    for line in open(
                        os.path.join(
                            data_path, "socialiqa-train-dev", "socialiqa-train-dev", f"{split_mapping[split]}.jsonl"
                        )
                    )
                ]
                labels = [
                    int(line.strip())
                    for line in open(
                        os.path.join(
                            data_path,
                            "socialiqa-train-dev",
                            "socialiqa-train-dev",
                            f"{split_mapping[split]}-labels.lst",
                        )
                    )
                ]
                assert len(data) == len(labels)
                for item, label in zip(data, labels):
                    item["label"] = label
                with open(os.path.join(data_path, f"siqa_{split}.jsonl"), "w") as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def load_dataset(self) -> List[List[str]]:
        data_path = os.path.join(self.output_path, "data", self.dataset)

        if self.dataset == "hellaswag":
            split_to_file = {
                split: os.path.join(data_path, f"hellaswag_{split}.jsonl") for split in ["train", "val"]
            }  # Ignore HellaSwag test set because no label information
            item_process_func = self.process_hellaswag_item
        elif self.dataset == "openbookqa":
            split_to_file = {
                split: os.path.join(data_path, "OpenBookQA-V1-Sep2018", "Data", "Main", f"{split}.jsonl")
                for split in ["train", "test"]
            }
            item_process_func = self.process_openbookqa_item
        elif self.dataset == "commonsenseqa":
            split_to_file = {
                split: os.path.join(data_path, f"commonsenseqa_{split}.jsonl") for split in ["train", "val"]
            }  # Ignore CommonSenseQA test set because no label information
            item_process_func = self.process_commonsenseqa_item
        elif self.dataset == "piqa":
            split_to_file = {
                split: os.path.join(data_path, f"piqa_{split}.jsonl") for split in ["train", "val"]
            }  # Ignore PIQA test set because no label information
            item_process_func = self.process_piqa_item
        elif self.dataset == "siqa":
            split_to_file = {
                split: os.path.join(data_path, f"siqa_{split}.jsonl") for split in ["train", "val"]
            }  # SIQA has no available test set
            item_process_func = self.process_siqa_item
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        data = []
        for split in split_to_file:
            file_path = split_to_file[split]
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

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
            "train": TRAIN_SPLIT,
            "val": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        instances: List[Instance] = []

        def answer_to_reference(answer: str) -> Reference:
            return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

        for question_id, (question, answers, correct_answer, split) in enumerate(data):
            instance = Instance(
                input=Input(text=question),
                references=list(map(answer_to_reference, answers)),
                split=splits[split],
            )
            instances.append(instance)
        return instances
