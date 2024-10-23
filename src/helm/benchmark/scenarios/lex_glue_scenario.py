import random
from pathlib import Path
from typing import List, Any

import datasets
from datasets import load_dataset

from helm.benchmark.scenarios.lextreme_scenario import TaskType
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    Input,
    Output,
)

ECTHR_A = "ecthr_a"
ECTHR_B = "ecthr_b"
SCOTUS = "scotus"
EURLEX = "eurlex"
LEDGAR = "ledgar"
UNFAIR_TOS = "unfair_tos"
CASE_HOLD = "case_hold"

TASK_CODE_MAPPING = {
    ECTHR_A: TaskType.MLTC,
    ECTHR_B: TaskType.MLTC,
    SCOTUS: TaskType.SLTC,
    EURLEX: TaskType.MLTC,
    LEDGAR: TaskType.SLTC,
    UNFAIR_TOS: TaskType.MLTC,
    CASE_HOLD: TaskType.QA,
}


def get_lex_glue_task_type(subset):
    return TASK_CODE_MAPPING[subset]


TASK_MAX_TRAIN_INSTANCES_MAPPING = {
    ECTHR_A: 1,  # ~ max 4096 tokens
    ECTHR_B: 1,  # ~ max 4096 tokens
    SCOTUS: 1,  # ~ max 8192 tokens
    EURLEX: 5,  # ~ max 512 tokens
    LEDGAR: 5,  # ~ max 512 tokens
    UNFAIR_TOS: 5,  # ~ max 128 tokens
    CASE_HOLD: 5,  # ~ max 512 tokens
}


def get_lex_glue_max_train_instances(subset):
    return TASK_MAX_TRAIN_INSTANCES_MAPPING[subset]


TASK_MAX_TOKENS_MAPPING = {
    ECTHR_A: 20,  # sequence of numbers
    ECTHR_B: 20,  # sequence of numbers
    SCOTUS: 5,  # one number
    EURLEX: 20,  # sequence of numbers
    LEDGAR: 20,  # multiple words
    UNFAIR_TOS: 20,  # sequence of numbers
    CASE_HOLD: 5,  # one number
}


def get_lex_glue_max_tokens(subset):
    return TASK_MAX_TOKENS_MAPPING[subset]


INSTRUCTIONS = {
    ECTHR_A: "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
    "Predict the articles of the ECtHR that were violated (if any) out of the following: "
    "0: Article 2, "
    "1: Article 3, "
    "2: Article 5, "
    "3: Article 6, "
    "4: Article 8, "
    "5: Article 9, "
    "6: Article 10, "
    "7: Article 11, "
    "8: Article 14, "
    "9: Article 1 of Protocol 1. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    ECTHR_B: "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
    "Predict the articles of ECtHR that were allegedly violated (considered by the court) out of the following:"
    "0: Article 2, "
    "1: Article 3, "
    "2: Article 5, "
    "3: Article 6, "
    "4: Article 8, "
    "5: Article 9, "
    "6: Article 10, "
    "7: Article 11, "
    "8: Article 14, "
    "9: Article 1 of Protocol 1. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    SCOTUS: "In this task, you are given a case heard at the Supreme Court of the United States (SCOTUS). "
    "Predict the relevant issue area out of the following: "
    "0: Criminal Procedure, "
    "1: Civil Rights, "
    "2: First Amendment, "
    "3: Due Process, "
    "4: Privacy, "
    "5: Attorneys, "
    "6: Unions, "
    "7: Economic Activity, "
    "8: Judicial Power, "
    "9: Federalism, "
    "10: Interstate Relations, "
    "11: Federal Taxation, "
    "12: Miscellaneous, "
    "13: Private Action.",
    EURLEX: "In this task, you are given an EU law document published in the EUR-Lex portal. "
    "Predict the relevant EuroVoc concepts. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    LEDGAR: "In this task, you are given a contract provision "
    "from contracts obtained from US Securities and Exchange Commission (SEC) filings."
    "Predict the main topic. ",
    UNFAIR_TOS: "In this task, you are given a sentence "
    "from a Terms of Service (ToS) document from online platforms. "
    "Predict the types of unfair contractual terms out of the following: "
    "0: Limitation of liability, "
    "1: Unilateral termination, "
    "2: Unilateral change, "
    "3: Content removal, "
    "4: Contract by using, "
    "5: Choice of law, "
    "6: Jurisdiction, "
    "7: Arbitration. "
    "If there is no label reply n/a, if there are multiple labels specify all of them separated by a comma.",
    CASE_HOLD: "In this task, you are given an excerpt from a court decision, "
    "containing a reference to a particular case, while the holding statement is masked out. "
    "Predict the index of the holding statement fitting in the context at <HOLDING> from a selection of five choices.",
}


def get_lex_glue_instructions(subset):
    return INSTRUCTIONS[subset]


class LexGLUEScenario(Scenario):
    """
    Inspired by the recent widespread use of the GLUE multi-task benchmark NLP dataset (Wang et al., 2018),
    the subsequent more difficult SuperGLUE (Wang et al., 2019),
    other previous multi-task NLP benchmarks (Conneau and Kiela, 2018; McCann et al., 2018),
    and similar initiatives in other domains (Peng et al., 2019),
    we introduce the Legal General Language Understanding Evaluation (LexGLUE) benchmark,
    a benchmark dataset to evaluate the performance of NLP methods in legal tasks.
    LexGLUE is based on seven existing legal NLP datasets, selected using criteria largely from SuperGLUE.
    Find more information on the dataset here: https://huggingface.co/datasets/lex_glue

    We prompt models using the following format (example for unfair_tos)

        <sentence>
        Unfair Contractual Term Type:

        Target completion:
            <sentence> (<sentence>:"Limitation of liability", "Unilateral termination", "Unilateral change",
                        "Content removal", "Contract by using", "Choice of law", "Jurisdiction", "Arbitration")

    Using an example from the training dataset, we have

    ```
    "tinder may terminate your account at any time without notice if it believes that you have violated this agreement."

    Unfair Contractual Term Type:
    Target completion:
        "Unilateral change"
    ```

    """

    name = "lex_glue"
    description = "A Benchmark Dataset for Legal Language Understanding in English."
    tags = ["single_label_text_classification", "multi_label_text_classification", "question_answering"]

    # Mapping from HELM splits to HF splits
    splits_mapping = {
        TRAIN_SPLIT: datasets.Split.TRAIN,
        VALID_SPLIT: datasets.Split.VALIDATION,
        TEST_SPLIT: datasets.Split.TEST,
    }

    dataset_name = "lex_glue"
    max_number_of_wrong_answers = 30

    def __init__(self, subset: str):
        super().__init__()
        assert subset in list(TASK_CODE_MAPPING.keys()) + ["all"], f"Unknown subset: {subset}"
        self.subsets = [subset] if subset != "all" else list(TASK_CODE_MAPPING.keys())
        self.random: random.Random = random.Random(42)

    def get_instances_for_subset(self, config: str, output_path: str) -> List[Instance]:
        task_code = TASK_CODE_MAPPING[config]
        # Load dataset
        cache_dir = str(Path(output_path) / "data")
        dataset: Any = load_dataset(self.dataset_name, config, cache_dir=cache_dir)

        if task_code in [TaskType.SLTC, TaskType.QA]:
            class_label = dataset["train"].features["label"]
            label_classes = class_label.names
        elif task_code == TaskType.MLTC:
            # construct the label classes
            label_classes = set()
            for split in self.splits_mapping.values():
                for example in dataset[split]:
                    label_classes |= set(example["labels"])  # add all new labels to the set
            label_classes = sorted(list(map(str, label_classes)))  # convert everything to a string

        def generate_instance(example, split: str):
            # get correct labels
            if task_code in [TaskType.SLTC, TaskType.QA]:
                correct_label = class_label.int2str(example["label"])  # get label name for correct label
                correct_labels = correct_label if isinstance(correct_label, list) else [correct_label]
            elif task_code == TaskType.MLTC:
                correct_labels = list(map(str, example["labels"]))  # here we don't have any mapping to label names

            # construct wrong references
            wrong_references = []
            for label_name in label_classes:
                if label_name not in correct_labels:
                    wrong_reference = Reference(output=Output(label_name), tags=[])  # Wrong output
                    wrong_references.append(wrong_reference)

            wrong_references = reduce_wrong_reference_count(wrong_references)

            # construct correct references and input
            if task_code in [TaskType.SLTC, TaskType.MLTC]:
                input_text = example["text"]
                if "ecthr" in config:
                    input_text = " ".join(input_text)
            elif task_code == TaskType.QA:
                endings = [f"{i}: {end}" for i, end in enumerate(example["endings"])]
                input_text = example["context"] + " Holdings: " + " ".join(endings)

            # construct correct references
            correct_references = [
                Reference(output=Output(correct_label), tags=[CORRECT_TAG]) for correct_label in correct_labels
            ]  # for MLTC we have multiple correct ones
            return Instance(input=Input(input_text), references=wrong_references + correct_references, split=split)

        def reduce_wrong_reference_count(wrong_references):
            self.random.shuffle(wrong_references)  # shuffle wrong references
            if len(wrong_references) > self.max_number_of_wrong_answers:
                # if there are too many wrong references, only take a subset
                wrong_references = wrong_references[: self.max_number_of_wrong_answers]
            return wrong_references

        def generate_instances(split: str):
            split_dataset = dataset[self.splits_mapping[split]]
            return [generate_instance(example, split) for example in split_dataset]

        return generate_instances(TRAIN_SPLIT) + generate_instances(VALID_SPLIT) + generate_instances(TEST_SPLIT)

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []
        for subset in self.subsets:
            instances.extend(self.get_instances_for_subset(subset, output_path))
        return instances
