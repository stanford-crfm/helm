import json
import os
from typing import List

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
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


_SPLIT_TRANSLATION = {
    "train": TRAIN_SPLIT,
    "val": VALID_SPLIT,
    "test": TEST_SPLIT,
}


def _make_instance(question: str, answers: List[str], correct_answer: str, split: str):
    references = []
    for answer in answers:
        references.append(Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else []))
    return Instance(
        input=Input(text=question),
        references=references,
        split=_SPLIT_TRANSLATION[split],
    )


class HellaSwagScenario(Scenario):
    name = "hellaswag"
    description = "Benchmark from https://arxiv.org/pdf/1905.07830.pdf."
    tags = ["knowledge", "multiple_choice"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances = []
        base_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_{}.jsonl"
        # Ignore HellaSwag test set because no label information
        for split in ["train", "val"]:
            file_path = os.path.join(data_path, f"hellaswag_{split}.jsonl")
            ensure_file_downloaded(
                source_url=base_url.format(split),
                target_path=file_path,
            )
            hlog(f"Reading {file_path}")
            with open(file_path) as f:
                for line in f:
                    item = json.loads(line)
                    instances.append(self.json_to_instance(item, split))
        return instances

    @staticmethod
    def json_to_instance(item, split) -> Instance:
        ctx_b_fixed = item["ctx_b"][0].upper() + item["ctx_b"][1:] if len(item["ctx_b"]) != 0 else ""

        question = f"{item['activity_label']}: {item['ctx_a']} {ctx_b_fixed}"
        answers = item["endings"]
        correct_answer = answers[item["label"]]

        assert len(answers) == 4
        return _make_instance(question=question, answers=answers, correct_answer=correct_answer, split=split)


class OpenBookQA(Scenario):
    name = "openbookqa"
    description = "Benchmark from https://aclanthology.org/D18-1260.pdf."
    tags = ["knowledge", "multiple_choice"]

    def get_instances(self, output_path: str):
        # Download the raw data
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        ensure_file_downloaded(
            source_url="https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip",
            target_path=os.path.join(data_path, "OpenBookQA-V1-Sep2018"),
            unpack=True,
            unpack_type="unzip",
        )

        instances = []
        for split in ["train", "test"]:
            file_path = os.path.join(data_path, "OpenBookQA-V1-Sep2018", "Data", "Main", f"{split}.jsonl")
            hlog(f"Reading {file_path}")
            with open(file_path) as f:
                for line in f:
                    item = json.loads(line)
                    instances.append(self.json_to_instance(item, split))
        return instances

    @staticmethod
    def json_to_instance(item, split) -> Instance:
        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3}

        question = item["question"]["stem"]
        answers = [answer["text"] for answer in item["question"]["choices"]]
        correct_choice = letter2idx[item["answerKey"]]
        correct_answer = answers[correct_choice]

        assert len(answers) == 4
        assert item["question"]["choices"][correct_choice]["label"] == item["answerKey"]
        return _make_instance(question=question, answers=answers, correct_answer=correct_answer, split=split)


class CommonSenseQAScenario(Scenario):
    name = "commonsenseqa"
    description = "Benchmark from https://arxiv.org/pdf/1811.00937.pdf."
    tags = ["knowledge", "multiple_choice"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances = []
        base_url = "https://s3.amazonaws.com/commensenseqa/{}_rand_split.jsonl"
        # Ignore CommonSenseQA test set because no label information
        split_mapping = {"train": "train", "val": "dev"}
        for split in ["train", "val"]:
            file_path = os.path.join(data_path, f"commonsenseqa_{split}.jsonl")
            ensure_file_downloaded(
                source_url=base_url.format(split_mapping[split]),
                target_path=file_path,
            )
            hlog(f"Reading {file_path}")
            with open(file_path) as f:
                for line in f:
                    item = json.loads(line)
                    instances.append(self.json_to_instance(item, split))
        return instances

    @staticmethod
    def json_to_instance(item, split) -> Instance:
        # Note: question concept field is not used: item["question"]["question_concept"]
        letter2idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        question = item["question"]["stem"]
        answers = [answer["text"] for answer in item["question"]["choices"]]
        correct_choice = letter2idx[item["answerKey"]]
        correct_answer = answers[correct_choice]

        assert len(answers) == 5
        assert item["question"]["choices"][correct_choice]["label"] == item["answerKey"]
        return _make_instance(question, answers, correct_answer, split)


class PiqaScenario(Scenario):
    name = "piqa"
    description = "Benchmark from https://arxiv.org/pdf/1911.11641.pdf."
    tags = ["knowledge", "multiple_choice"]

    def get_instances(self, output_path: str):
        # Download the raw data
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        url = "https://yonatanbisk.com/piqa/data/{}"
        # TODO The source actually uses TRAIN_SPLIT and VALID_SPLIT, so consider skipping "val".
        split_mapping = {"train": "train", "val": "valid"}
        instances = []
        # Ignore PIQA test set because no label information
        for split in ["train", "val"]:
            ensure_file_downloaded(
                source_url=url.format(f"{split_mapping[split]}.jsonl"),
                target_path=os.path.join(data_path, f"piqa_{split}.jsonl"),
            )
            ensure_file_downloaded(
                source_url=url.format(f"{split_mapping[split]}-labels.lst"),
                target_path=os.path.join(data_path, f"piqa_{split}_labels.lst"),
            )
            data = [json.loads(line) for line in open(os.path.join(data_path, f"piqa_{split}.jsonl"))]
            labels = [int(line.strip()) for line in open(os.path.join(data_path, f"piqa_{split}_labels.lst"))]
            assert len(data) == len(labels)
            for item, label in zip(data, labels):
                instances.append(self.json_to_instance(item, label, split))
        return instances

    @staticmethod
    def json_to_instance(item, label: int, split: str):
        question = item["goal"]
        answers = [item["sol1"], item["sol2"]]
        correct_choice = label
        correct_answer = answers[correct_choice]

        assert len(item) == 3
        assert correct_choice in [0, 1]
        return _make_instance(question, answers, correct_answer, split)


class SiqaScenario(Scenario):
    name = "siqa"
    description = "Benchmark from https://arxiv.org/pdf/1904.09728.pdf."
    tags = ["knowledge", "multiple_choice"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        ensure_file_downloaded(
            source_url="https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
            target_path=os.path.join(data_path, "socialiqa-train-dev"),
            unpack=True,
            unpack_type="unzip",
        )
        # TODO The source doesn't follow the standard naming for 'val', so maybe can skip _SPLIT_TRANSLATION.
        split_mapping = {"train": "train", "val": "dev"}
        instances = []
        # SIQA has no available test set
        for split in ["train", "val"]:
            base_path = os.path.join(data_path, "socialiqa-train-dev", "socialiqa-train-dev", f"{split_mapping[split]}")
            data = [json.loads(line) for line in open(base_path + ".jsonl")]
            labels = [int(line.strip()) for line in open(base_path + "-labels.lst")]
            assert len(data) == len(labels)

            for item, label in zip(data, labels):
                instances.append(self.json_to_instance(item, label, split))
        return instances

    @staticmethod
    def json_to_instance(item, label, split) -> Instance:
        question = f"{item['context']} {item['question']}"
        answers = [item["answerA"], item["answerB"], item["answerC"]]
        correct_choice = label - 1
        correct_answer = answers[correct_choice]

        assert len(item) == 5
        assert correct_choice in [0, 1, 2]
        return _make_instance(question, answers, correct_answer, split)
