import os
from typing import List

import datasets

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class MMMLUScenario(Scenario):
    """Multilingual Massive Multitask Language Understanding (MMMLU) by OpenAI

    The MMLU is a widely recognized benchmark of general knowledge attained
    by AI models. It covers a broad range of topics from 57 different categories,
    covering elementary-level knowledge up to advanced professional subjects like
    law, physics, history, and computer science.

    MMMLU is a translation of MMLU’s test set into 14 languages using professional
    human translators. Relying on human translators for this evaluation increases
    confidence in the accuracy of the translations, especially for low-resource
    languages like Yoruba.

    The Massive Multitask Language Understanding benchmark from this paper:

    - https://arxiv.org/pdf/2009.03300.pdf

    The MMMLU dataset is from here:

    - https://huggingface.co/datasets/openai/MMMLU
    """

    name = "mmmlu"
    description = "Multilingual Massive Multitask Language Understanding"
    tags = ["knowledge", "multiple_choice"]

    OPTIONS = ["A", "B", "C", "D"]

    def __init__(self, locale: str, subject: str):
        super().__init__()
        self.locale: str = locale
        self.subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "openai/MMMLU",
            self.locale,
            revision="325a01dc3e173cac1578df94120499aaca2e2504",
            cache_dir=cache_dir,
            split="test",
        )
        assert isinstance(dataset, datasets.Dataset)

        # Read all instances
        instances: List[Instance] = []
        for row_index, row in enumerate(dataset):
            if self.subject != "all" and row["Subject"] != self.subject:
                continue
            input = Input(text=row["Question"])
            references: List[Reference] = []
            for option in self.OPTIONS:
                references.append(
                    Reference(
                        output=Output(text=row[option]),
                        tags=[CORRECT_TAG] if option == row["Answer"] else [],
                    )
                )
            instance = Instance(
                id=f"id{row_index}",
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
