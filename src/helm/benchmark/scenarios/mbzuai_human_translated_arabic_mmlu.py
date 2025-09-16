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


class MBZUAIHumanTranslatedArabicMMLUScenario(Scenario):
    """MBZUAI Human-Translated Arabic MMLU

    A translation from MBZUAI by human translators of the Massive Multitask Language Understanding benchmark from this paper:

    - https://arxiv.org/pdf/2009.03300.pdf
    """  # noqa: E501

    name = "mbzuai_human_translated_arabic_mmlu"
    description = (
        "A translation from MBZUAI by human translators of the Massive Multitask Language Understanding benchmark"
    )
    tags = ["knowledge", "multiple_choice"]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "MBZUAI/human_translated_arabic_mmlu",
            self.subject,
            revision="5ed7830fd678cfa6f2d7f0a1a13a4e1a1fa422ac",
            cache_dir=cache_dir,
            split="test",
        )
        assert isinstance(dataset, datasets.Dataset)

        # Read all instances
        instances: List[Instance] = []
        for row_index, row in enumerate(dataset):
            input = Input(text=row["question"])
            references: List[Reference] = []
            for choice_index, choice in enumerate(row["choices"]):
                references.append(
                    Reference(
                        output=Output(text=choice),
                        tags=[CORRECT_TAG] if choice_index == row["answer"] else [],
                    )
                )
            instance = Instance(
                id=f"id-{self.subject}-{row_index}",
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
