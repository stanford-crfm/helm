import datasets
import os
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


class Math500Scenario(Scenario):
    """MATH-500 benchmark from https://huggingface.co/datasets/HuggingFaceH4/MATH-500."""

    name = "math500"
    description = "MATH-500: competition-level mathematics benchmark"
    tags = ["math"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", cache_dir=cache_dir, split="test")
        assert isinstance(dataset, datasets.Dataset)

        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["problem"])
            instance = Instance(
                input=input,
                references=[Reference(Output(text=row["answer"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name=self.name,
            display_name="MATH-500",
            description=self.description,
            main_metric="math_accuracy",
            main_split="test",
            taxonomy=TaxonomyInfo(
                task="mathematics",
                what="competition-level mathematics problems",
                who="competition authors",
                when="2024",
                language="English",
            ),
        )

