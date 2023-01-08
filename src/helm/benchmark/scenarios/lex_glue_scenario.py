import random
from pathlib import Path
from typing import List

import datasets
from datasets import load_dataset

from .lextreme_scenario import TaskType
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT

TASK_CODE_MAPPING = {
    'ecthr_a': TaskType.MLTC,
    'ecthr_b': TaskType.MLTC,
    'scotus': TaskType.SLTC,
    'eurlex': TaskType.MLTC,
    'ledgar': TaskType.SLTC,
    'unfair_tos': TaskType.MLTC,
    'case_hold': TaskType.QA,
}

TASK_MAX_TRAIN_INSTANCES_MAPPING = {
    'ecthr_a': 1,  # ~ max 4096 tokens
    'ecthr_b': 1,  # ~ max 4096 tokens
    'scotus': 1,  # ~ max 8192 tokens
    'eurlex': 5,  # ~ max 512 tokens
    'ledgar': 5,  # ~ max 512 tokens
    'unfair_tos': 5,  # ~ max 128 tokens
    'case_hold': 5,  # ~ max 512 tokens
}


def get_lex_glue_max_train_instances(subset):
    return TASK_MAX_TRAIN_INSTANCES_MAPPING[subset]


TASK_MAX_TOKENS_MAPPING = {
    'ecthr_a': 20,  # sequence of numbers
    'ecthr_b': 20,  # sequence of numbers
    'scotus': 5,  # one number
    'eurlex': 20,  # sequence of numbers
    'ledgar': 20,  # multiple words
    'unfair_tos': 20,  # sequence of numbers
    'case_hold': 5,  # one number
}


def get_lex_glue_max_tokens(subset):
    return TASK_MAX_TOKENS_MAPPING[subset]


INSTRUCTIONS = {
    "ecthr_a": "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
               "Predict the articles of the ECtHR that were violated (if any).",
    "ecthr_b": "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
               "Predict the articles of ECtHR that were allegedly violated (considered by the court).",
    "scotus": "In this task, you are given a case heard at the Supreme Court of the United States (SCOTUS). "
              "Predict the relevant issue area.",
    "eurlex": "In this task, you are given an EU law document published in the EUR-Lex portal. "
              "Predict the relevant EuroVoc concepts.",
    "ledgar": "In this task, you are given a contract provision from contracts obtained from US Securities and Exchange Commission (SEC) filings."
              "Predict the main topic.",
    "unfair_tos": "In this task, you are given a sentence from a Terms of Service (ToS) document from on-line platforms. "
                  "Predict the types of unfair contractual terms",
    "case_hold": "In this task, you are given an excerpt from a court decision, containing a reference to a particular case, while the holding statement is masked out. "
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
            <sentence> (<sentence>:"Limitation of liability", "Unilateral termination", "Unilateral change", "Content removal", "Contract by using", "Choice of law", "Jurisdiction", "Arbitration")

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
        TEST_SPLIT: datasets.Split.TEST
    }

    dataset_name = "lex_glue"
    max_number_of_wrong_answers = 30
    mltc_no_label_name = 'No Label'
    delimiter = "|"  # we choose the pipe as a delimiter because it is very unlikely to occur in the data

    def __init__(self, subset: str):
        assert subset in list(TASK_CODE_MAPPING.keys()) + ["all"], f"Unknown subset: {subset}"
        self.subsets = [subset] if subset != "all" else list(TASK_CODE_MAPPING.keys())
        self.random: random.Random = random.Random(42)

    def get_instances_for_subset(self, config: str) -> List[Instance]:
        task_code = TASK_CODE_MAPPING[config]
        # Load dataset
        cache_dir = str(Path(self.output_path) / "data")
        dataset = load_dataset(self.dataset_name, config, cache_dir=cache_dir)

        if task_code in [TaskType.SLTC, TaskType.QA]:
            class_label = dataset['train'].features["label"]
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
                correct_label = class_label.int2str(example['label'])  # get label name for correct label
                correct_labels = correct_label if isinstance(correct_label, list) else [correct_label]
            elif task_code == TaskType.MLTC:
                correct_labels = list(map(str, example['labels']))  # here we don't have any mapping to label names

            # construct wrong references
            wrong_references = []
            for label_name in label_classes:
                if label_name not in correct_labels:
                    wrong_reference = Reference(output=label_name, tags=[])  # Wrong output
                    wrong_references.append(wrong_reference)

            wrong_references = reduce_wrong_reference_count(wrong_references)

            if task_code == TaskType.MLTC:  # special case for multilabel classification tasks
                if correct_labels:  # if we have a correct label
                    # add the no_label to the wrong references
                    # IMPORTANT: add it after the reduce_wrong_reference_count call, to make sure the no label is always there
                    wrong_references.append(Reference(output=self.mltc_no_label_name, tags=[]))
                else:  # if we don't have a correct label
                    # add the no_label to the correct labels
                    correct_labels = [self.mltc_no_label_name]

            # construct correct references and input
            if task_code in [TaskType.SLTC, TaskType.MLTC]:
                input_text = example['text']
                if 'ecthr' in config:
                    input_text = " ".join(input_text)
            elif task_code == TaskType.QA:
                endings = [f"{i}: {end}" for i, end in enumerate(example['endings'])]
                input_text = example['context'] + " Holdings: " + " ".join(endings)

            # construct correct references
            correct_references = [Reference(output=correct_label, tags=[CORRECT_TAG])
                                  for correct_label in correct_labels]  # for MLTC we have multiple correct ones
            return Instance(input=input_text, references=wrong_references + correct_references, split=split)

        def reduce_wrong_reference_count(wrong_references):
            self.random.shuffle(wrong_references)  # shuffle wrong references
            if len(wrong_references) > self.max_number_of_wrong_answers:
                # if there are too many wrong references, only take a subset
                wrong_references = wrong_references[:self.max_number_of_wrong_answers]
            return wrong_references

        def generate_instances(split: str):
            split_dataset = dataset[self.splits_mapping[split]]
            return [generate_instance(example, split) for example in split_dataset]

        return generate_instances(TRAIN_SPLIT) + generate_instances(VALID_SPLIT) + generate_instances(TEST_SPLIT)

    def get_instances(self) -> List[Instance]:
        instances = []
        for subset in self.subsets:
            instances.extend(self.get_instances_for_subset(subset))
        return instances
