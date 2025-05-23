from abc import abstractmethod
from typing import Dict, List, Tuple, Optional

import random
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput,
    Input,
    Output,
)


class MELTClosedBookQAScenario(Scenario):
    name = "melt_closed_book_qa"
    description = "Closed Book Question Answering scenario."
    tags = ["question_answering"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        subset: Optional[str] = None,
        splits: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.revision = revision
        self.splits = splits

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )
        for dataset_split_name, helm_split_name in splits.items():

            for sample in dataset[dataset_split_name]:
                instance = Instance(
                    input=Input(text=sample["question"]),
                    references=[Reference(Output(text=sample["answer"]), tags=[CORRECT_TAG])],
                    split=helm_split_name,
                )
                instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        if self.splits is None:
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        else:
            splits = {}
            if "train" in self.splits:
                splits[self.splits[TRAIN_SPLIT]] = TRAIN_SPLIT
            if "validation" in self.splits:
                splits[self.splits[VALID_SPLIT]] = VALID_SPLIT
            if "test" in self.splits:
                splits[self.splits[TEST_SPLIT]] = TEST_SPLIT

        instances: List[Instance] = self.get_instances_for_splits(splits=splits)
        return instances


class MELTKnowledgeZaloScenario(MELTClosedBookQAScenario):
    """
    Scenario for the Zalo dataset.
    """

    name = "melt_knowledge_zalo"
    description = "Zalo dataset for closed-book question answering."
    tags = ["question_answering", "knowledge"]

    def __init__(self):
        super().__init__(
            dataset_name="ura-hcmut/zalo_e2eqa",
            revision="63494521f4de949bfa57a5f0b79bc3ee47e635ad",
            splits={
                TRAIN_SPLIT: "train",
                TEST_SPLIT: "test",
            },
        )


class MELTMultipleChoiceQAScenario(Scenario):
    name = "melt_multiple_choice_qa"
    description = "Multiple Choice Question Answering scenario."
    tags = ["question_answering"]

    def __init__(
        self,
        dataset_name: str,
        revision: str,
        subset: Optional[str] = None,
        splits: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the question answering scenario.

        Args:
            dataset_name: The name of the dataset.
            revision: The revision of the dataset to use.
            subset: The subset of the dataset to use. Defaults to "".
            splits: The splits to use for the dataset. Defaults to None.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.revision = revision
        self.splits = splits

    @abstractmethod
    def process_example(self, sample: dict) -> Tuple[Input, List[Reference]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """
        pass

    def get_instances_for_splits(self, splits: Dict[str, str]) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            splits (dict): Which splits to partition the data into.
        Returns:
            List[Instance]: Instances from the file for the specified split.
        """
        instances: List[Instance] = []
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            revision=self.revision,
            trust_remote_code=True,
        )
        for dataset_split_name, helm_split_name in splits.items():
            for sample in dataset[dataset_split_name]:
                inputs, references = self.process_example(sample)
                instance = Instance(
                    input=inputs,
                    references=references,
                    split=helm_split_name,
                )
                instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        if self.splits is None:
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        else:
            splits = {}
            if "train" in self.splits:
                splits[self.splits[TRAIN_SPLIT]] = TRAIN_SPLIT
            if "validation" in self.splits:
                splits[self.splits[VALID_SPLIT]] = VALID_SPLIT
            if "test" in self.splits:
                splits[self.splits[TEST_SPLIT]] = TEST_SPLIT

        instances: List[Instance] = self.get_instances_for_splits(splits=splits)
        return instances


class MELTKnowledgeViMMRCScenario(MELTMultipleChoiceQAScenario):
    """
    Scenario for the ViMMRC dataset.
    """

    name = "melt_knowledge_vimmrc"
    description = "ViMMRC dataset for multiple choice question answering."
    tags = ["question_answering", "knowledge"]

    def __init__(self, randomize_order: bool = False):
        super().__init__(
            dataset_name="ura-hcmut/ViMMRC",
            revision="fe68800e37aaa84d80b1d93466b36c3fa60d8bcb",
            splits={
                TRAIN_SPLIT: "train",
                VALID_SPLIT: "validation",
                TEST_SPLIT: "test",
            },
        )
        self.randomize_order = randomize_order
        self.correct_answer_mapping = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }
        random.seed(42)

    def process_example(self, sample: dict) -> Tuple[Input, List[Reference]]:
        """
        Given an sample from the dataset, create the input text and
        list of answers for the instance.
        """
        inputs = PassageQuestionInput(
            passage=sample["article"],
            passage_prefix="Ngữ cảnh: ",
            question=sample["question"],
            question_prefix="Câu hỏi: ",
            separator="\n\n",
        )

        correct_idx = self.correct_answer_mapping[sample["answer"]]
        references = []
        for idx, answer in enumerate(eval(sample["options"])):
            if idx == correct_idx:
                tags = [CORRECT_TAG]
            else:
                tags = []

            references.append(Reference(Output(text=answer), tags=tags))

        if self.randomize_order:
            random.shuffle(references)

        return inputs, references
