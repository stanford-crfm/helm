import datasets
import os
import re
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    Input,
    Output,
    CORRECT_TAG,
    ScenarioMetadata,
)
from helm.common.general import ensure_directory_exists


class AIMEScenario(Scenario):
    """AIME 2024 benchmark from https://huggingface.co/datasets/math-ai/aime24."""

    name = "aime"
    description = "AIME 2024 mathematics problems"
    tags = ["math"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset("math-ai/aime24", cache_dir=cache_dir, split="test")
        assert isinstance(dataset, datasets.Dataset)

        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["problem"])
            solution: str = row["solution"]
            instance = Instance(
                input=input,
                references=[Reference(Output(text=solution), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name=self.name,
            display_name="AIME 2024",
            description=self.description,
            main_metric="math_accuracy",
            main_split="test",
            taxonomy=TaxonomyInfo(
                task="mathematics",
                what="American Invitational Mathematics Examination 2024 problems",
                who="competition authors",
                when="2024",
                language="English",
            ),
        )


class AIME25Scenario(Scenario):
    """AIME 2025 benchmark from https://huggingface.co/datasets/opencompass/AIME2025."""

    name = "aime25"
    description = "AIME 2025 mathematics problems"
    tags = ["math"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        dataset_i = datasets.load_dataset("opencompass/AIME2025", "AIME2025-I", cache_dir=cache_dir, split="test")
        dataset_ii = datasets.load_dataset("opencompass/AIME2025", "AIME2025-II", cache_dir=cache_dir, split="test")
        assert isinstance(dataset_i, datasets.Dataset)
        assert isinstance(dataset_ii, datasets.Dataset)

        instances: List[Instance] = []
        for dataset in (dataset_i, dataset_ii):
            for row in dataset:
                input = Input(text=row["question"])
                answer = row["answer"]
                instance = Instance(
                    input=input,
                    references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
                instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name=self.name,
            display_name="AIME 2025",
            description=self.description,
            main_metric="math_accuracy",
            main_split="test",
            taxonomy=TaxonomyInfo(
                task="mathematics",
                what="American Invitational Mathematics Examination 2025 problems",
                who="competition authors",
                when="2025",
                language="English",
            ),
        )

