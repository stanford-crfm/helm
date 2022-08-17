import json
import os
from typing import List, Dict

from common.general import ensure_directory_exists, ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TEST_SPLIT


class BIGBenchScenario(Scenario):
    """
    The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to probe
    large language models and extrapolate their future capabilities. The more than 200 tasks included in
    BIG-bench.

    TODO: Finish description and cite -Tony
    """

    name = "big_bench"
    description = (
        "The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to "
        "probe large language models and extrapolate their future capabilities."
    )
    tags = ["general"]

    # TODO: update this link
    DOWNLOAD_URL: str = "https://raw.githubusercontent.com/UCSD-AI4H/BIG-bench/main/"

    def __init__(self, task: str, subtask: str):
        self.task: str = task
        self.subtask: str = subtask

    def get_instances(self) -> List[Instance]:
        data_path: str = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)
        dataset_path: str = os.path.join(data_path, self.task)
        ensure_file_downloaded(source_url=BIGBenchScenario.DOWNLOAD_URL, target_path=dataset_path)

        if self.subtask:
            dataset_path = os.path.join(dataset_path, self.subtask)
        dataset_path = os.path.join(dataset_path, "task.json")

        with open(dataset_path, "r") as f:
            task_definition: Dict = json.load(f)

        # Append the task-specific description to the `description`
        task_description: str = task_definition["description"]
        self.description = f"{self.description} Task description: {task_description}"

        # Read the examples for the task and construct `Instance`s
        instances: List[Instance] = []
        for example in task_definition["examples"]:
            # Each example has "input" and either "target" or "target_scores".
            # Build references
            references: List[Reference]
            if "target" in example:
                # All the outputs in "target" are correct e.g., `{"input": "1 + 1 = ", "target": ["two","2"]}`.
                references = [Reference(output, tags=[CORRECT_TAG]) for output in example["target"]]
            elif "target_scores" in example:
                # For "target_scores", BIG-bench compares target scores against the model's predicted probabilities.
                # For simplicity, `Reference`s with the highest score is considered correct.
                # TODO: double check this -Tony
                highest_score = max(example["target_scores"].values())
                references = [
                    Reference(output, tags=[CORRECT_TAG] if score == highest_score else [])
                    for output, score in example["target_scores"].items()
                ]
            else:
                raise ValueError(f"Invalid example: {example}")

            instances.append(Instance(example["input"], references, split=TEST_SPLIT))

        return instances
