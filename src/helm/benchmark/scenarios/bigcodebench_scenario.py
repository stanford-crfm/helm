import datasets
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    TEST_SPLIT,
    Input,
)
from helm.common.general import ensure_directory_exists


VERSIONS = ["v0.1.0_hf", "v0.1.1", "v0.1.2", "v0.1.3"]


class BigCodeBenchScenario(Scenario):
    """BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions

    BigCodeBench is an easy-to-use benchmark for solving practical and challenging tasks via code.
    It aims to evaluate the true programming capabilities of large language models (LLMs) in a more realistic setting.
    The benchmark is designed for HumanEval-like function-level code generation tasks,
    but with much more complex instructions and diverse function calls."""

    name = "bigcodebench"
    description = "Benchmarking Code Generation with Diverse Function Calls and Complex Instructions"
    tags = ["coding"]

    def __init__(self, version: str):
        super().__init__()
        assert version in VERSIONS, "Unknown version: {}".format(version)
        self.version = version

    def get_instances(self, output_path: str) -> List[Instance]:
        # Get BigCodeBench from HuggingFace
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "bigcode/bigcodebench",
            revision="057dd1a30dd73d4ed59cfbaaae049870491fa4d6",
            cache_dir=cache_dir,
            split=self.version,
        )
        assert isinstance(dataset, datasets.Dataset)

        # Read all instances
        instances: List[Instance] = []
        for row in dataset:
            id = row["task_id"]
            input = Input(text=row["instruct_prompt"])
            instance = Instance(
                id=id,
                input=input,
                references=[],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
