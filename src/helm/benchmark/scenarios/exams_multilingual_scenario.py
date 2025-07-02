import os
from typing import Dict, List

import datasets

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    VALID_SPLIT,
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    TRAIN_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.hierarchical_logger import hwarn


class EXAMSMultilingualScenario(Scenario):
    """EXAMS: A Multi-subject High School Examinations Dataset

    EXAMS is a benchmark dataset for multilingual and cross-lingual
    question answering from high school examinations. It consists of
    more than 24,000 high-quality high school exam questions in 16
    languages, covering 8 language families and 24 school subjects
    from Natural Sciences and Social Sciences, among others.

    - https://huggingface.co/datasets/mhardalov/exams
    - https://aclanthology.org/2020.emnlp-main.438/

    Note: Some dataset rows have the value '@' in the `answerKey` column.
    These rows will be ignored.

    ```
    @inproceedings{hardalov-etal-2020-exams,
        title = "{EXAMS}: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering",
        author = "Hardalov, Momchil  and
        Mihaylov, Todor  and
        Zlatkova, Dimitrina  and
        Dinkov, Yoan  and
        Koychev, Ivan  and
        Nakov, Preslav",
        editor = "Webber, Bonnie  and
        Cohn, Trevor  and
        He, Yulan  and
        Liu, Yang",
        booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2020.emnlp-main.438/",
        doi = "10.18653/v1/2020.emnlp-main.438",
        pages = "5427--5444",
        abstract = "We propose EXAMS {--} a new benchmark dataset for cross-lingual and multilingual question answering for high school examinations. We collected more than 24,000 high-quality high school exam questions in 16 languages, covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.EXAMS offers unique fine-grained evaluation framework across multiple languages and subjects, which allows precise analysis and comparison of the proposed models. We perform various experiments with existing top-performing multilingual pre-trained models and show that EXAMS offers multiple challenges that require multilingual knowledge and reasoning in multiple domains. We hope that EXAMS will enable researchers to explore challenging reasoning and knowledge transfer methods and pre-trained models for school question answering in various languages which was not possible by now. The data, code, pre-trained models, and evaluation are available at http://github.com/mhardalov/exams-qa."
    }```
    """  # noqa: E501

    name = "exams_multilingual"
    description = "EXAMS is a benchmark dataset for multilingual and cross-lingual question answering from high school examinations. "  # noqa: E501
    tags = ["knowledge", "multiple_choice"]

    CHOICES = ["A", "B", "C", "D", "E"]
    HF_SPLIT_TO_HELM_SPLIT = {"train": TRAIN_SPLIT, "test": TEST_SPLIT, "validation": VALID_SPLIT}

    def __init__(self, language: str, subject: str):
        super().__init__()
        self.language = language
        self.subject = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset_splits: Dict[str, datasets.Dataset] = datasets.load_dataset(
            "mhardalov/exams",
            "multilingual",
            revision="4ff10804abb3341f8815cacd778181177bba7edd",
            cache_dir=cache_dir,
        )

        # Read all instances
        instances: List[Instance] = []
        for split_name, dataset in dataset_splits.items():
            assert isinstance(dataset, datasets.Dataset)
            for row in dataset:
                question = row["question"]
                question_info = row["info"]
                if self.subject != "all" and question_info["subject"] != self.subject:
                    continue
                if self.language != "all" and question_info["language"] != self.language:
                    continue
                input = Input(text=question["stem"])
                references: List[Reference] = []
                if row["answerKey"] not in self.CHOICES:
                    hwarn(f"Invalid value in answerKey column in row: {row}")
                    continue
                correct_choice_index = ord(row["answerKey"]) - ord("A")
                for choice_index, choice_text in enumerate(question["choices"]["text"]):
                    references.append(
                        Reference(
                            output=Output(text=choice_text),
                            tags=[CORRECT_TAG] if choice_index == correct_choice_index else [],
                        )
                    )
                instance = Instance(
                    id=row["id"],
                    input=input,
                    references=references,
                    split=self.HF_SPLIT_TO_HELM_SPLIT[split_name],
                )
                instances.append(instance)

        return instances
