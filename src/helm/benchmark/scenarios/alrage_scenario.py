import os
from typing import List

import datasets

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)


class ALRAGEScenario(Scenario):
    """ALRAGE"""  # noqa: E501

    name = "alrage"
    description = "ALRAGE"
    tags = ["open-book question answering"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset: datasets.Dataset = datasets.load_dataset(
            "OALL/ALRAGE",
            revision="4827b2ed2436aea578e84d9bd4150b66ab8bbe0e",
            split="train",
            cache_dir=cache_dir,
        )

        # Read all instances
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=f"السؤال:\n{row['question']}\n\nالسياقات المقترحة:\n{row['candidates']}\n")
            references: List[Reference] = []
            references = [
                Reference(
                    output=Output(text=row["gold_answer"]),
                    tags=[CORRECT_TAG],
                )
            ]
            instance = Instance(
                id=row["id"],
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="alrage",
            display_name="ALRAGE",
            description="An Arabic language passage-based open-ended model-graded question answering benchmark that reflects retrieval-augmented generation use cases ([dataset](https://huggingface.co/datasets/OALL/ALRAGE))",
            taxonomy=TaxonomyInfo(
                task="openbook (RAG) open-ended question answering",
                what="passage-based question answering",
                when="book authors",
                who="before 2025",
                language="Arabic",
            ),
            main_metric="exact_match",
            main_split="test",
        )
