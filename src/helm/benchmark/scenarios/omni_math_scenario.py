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
    CORRECT_TAG,
)
from helm.common.general import ensure_directory_exists


class OmniMATHScenario(Scenario):
    """Omni-MATH: A Universal Olympiad Level Mathematic Benchmark for Large Language Models

    Omni-MATH is a comprehensive and challenging benchmark specifically designed to assess LLMs' mathematical
    reasoning at the Olympiad level. The dataset focuses exclusively on Olympiad mathematics and comprises a \
    vast collection of 4428 competition-level problems. These problems are meticulously categorized into 33 \
    (and potentially more) sub-domains and span across 10 distinct difficulty levels, enabling a nuanced \
    analysis of model performance across various mathematical disciplines and levels of complexity.."""

    name = "omni_math"
    description = "A Universal Olympiad Level Mathematic Benchmark for Large Language Models"
    tags = ["math"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Get Omni-MATH from HuggingFace
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "KbsdJames/Omni-MATH",
            revision="40ba231d8f16e29ecd40e6407e2c8640145a8f62",
            cache_dir=cache_dir,
            split="test",
        )
        assert isinstance(dataset, datasets.Dataset)

        # Read all instances
        instances: List[Instance] = []
        for idx, row in enumerate(dataset):

            input = Input(text=row["problem"])
            instance = Instance(
                input=input,
                references=[Reference(Output(text=row["answer"]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
