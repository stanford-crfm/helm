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
