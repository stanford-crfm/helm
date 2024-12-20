import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists


class AIRBench2024Scenario(Scenario):
    """AIRBench 2024

    Pre-publication: References will be added post-publication.

    AIRBench 2024 is a AI safety benchmark that aligns with emerging government
    regulations and company policies. It consists of 5,619 malicious prompts
    spanning categories of the regulation-based safety categories in the
    AIR 2024 safety taxonomy."""

    name = "air_bench_2024"
    description = (
        "AIRBench 2024 is a AI safety benchmark that aligns with "
        "emerging government regulations and company policies"
    )
    tags = ["safety"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        # TODO: Switch this to the production dataset when available.
        dataset = datasets.load_dataset(
            "stanford-crfm/air-bench-2024",
            split="test",
            cache_dir=cache_dir,
            revision="976d9fbb0f52bce704dcbef5bd7a2652e65f2a21",
        )
        instances: List[Instance] = []
        # TODO: Allow users to filter by category
        for row in dataset:
            input = Input(text=row["prompt"])
            # References are category ID, followed by level 2, 3 and 4 category names.
            references = [
                Reference(output=Output(text=row[column_name]), tags=[])
                for column_name in ["cate-idx", "l2-name", "l3-name", "l4-name"]
            ]
            instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)
        return instances
