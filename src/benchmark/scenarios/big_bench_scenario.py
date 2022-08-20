import json
import os
import random
from typing import List, Dict
from urllib.parse import urljoin

from common.general import ensure_directory_exists, ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT


class BIGBenchScenario(Scenario):
    """
    From Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models
    (https://arxiv.org/abs/2206.04615), the Beyond the Imitation Game Benchmark (BIG-bench) is a
    collaborative benchmark intended to probe large language models and extrapolate their future capabilities.
    More than 200 tasks included in BIG-bench.

    @misc{https://doi.org/10.48550/arxiv.2206.04615,
      doi = {10.48550/ARXIV.2206.04615},
      url = {https://arxiv.org/abs/2206.04615},
      author = {Srivastava et al.},
      title = {Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }
    """

    name = "big_bench"

    # Append the task-specific description after loading the task definition from BIG-bench
    description = (
        "The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to "
        "probe large language models and extrapolate their future capabilities."
    )

    # Will be updated after loading the task definition from BIG-bench
    tags: List[str] = []

    # Constants
    TASK_FILE_NAME: str = "task.json"
    MIN_TEST_EXAMPLES: int = 16

    @staticmethod
    def download_and_get_task(output_path: str, task: str, subtask: str) -> Dict:
        """
        Downloads the task folder from https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks
        if it doesn't already exist. Then, loads the BIG-bench task definition from task.json.
        """
        ensure_directory_exists(output_path)
        task_path: str = os.path.join(output_path, task)
        ensure_directory_exists(task_path)

        base_url: str = f"https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/{task}/"
        if subtask:
            base_url = urljoin(base_url, f"{subtask}/")
            task_path = os.path.join(task_path, subtask)
            ensure_directory_exists(task_path)

        target_path: str = os.path.join(task_path, BIGBenchScenario.TASK_FILE_NAME)
        ensure_file_downloaded(source_url=urljoin(base_url, BIGBenchScenario.TASK_FILE_NAME), target_path=target_path)
        with open(target_path, "r") as f:
            return json.load(f)

    def __init__(self, task: str, subtask: str):
        self.task: str = task
        self.subtask: str = subtask

    def get_instances(self) -> List[Instance]:
        """
        Construct `Instance`s using the examples from the BIG-bench task.
        """
        big_bench_task: Dict = BIGBenchScenario.download_and_get_task(self.output_path, self.task, self.subtask)

        # From https://github.com/google/BIG-bench/blob/main/docs/doc.md#json-schema,
        # "keywords", "description" and "examples" are all required fields for a BIG-bench task.

        # keywords: "A list of strings, where each string contains a separate keyword describing the task"
        self.tags = big_bench_task["keywords"]

        # description: "A plaintext description of the task, suitable for a non-expert to perform the task and
        #              potentially generate new examples."
        # Append the task-specific description from BIG-bench to the `description`.
        task_description: str = big_bench_task["description"]
        self.description = f"{self.description} Task description: {task_description}"

        # examples: "A list of dicts"
        examples: List[Dict] = big_bench_task["examples"]
        # Before splitting the data, shuffle the examples with a fixed seed for reproducibility.
        random.seed(0)
        random.shuffle(examples)

        # BIG-bench split the data according to
        # https://github.com/google/BIG-bench/blob/main/bigbench/bbseqio/README.md#splits:
        # all: This contains all the examples.
        # validation: This contains 20% of the examples or at least 16 examples.
        # train: All examples that are not in the validation split (generally 80% of the examples)
        # For few-shot eval, use the all split.
        #
        # I'm not sure what they mean by "for few-shot eval, use the all split."
        # We split the data as follows:
        # test: This contains 20% of the examples or at least 16 examples.
        # validation: Same size as the test split.
        # train: Remaining examples, not in the test and validation splits.
        total_examples = len(examples)
        num_test_examples: int = max(int(0.2 * total_examples), BIGBenchScenario.MIN_TEST_EXAMPLES)
        num_train_examples: int = total_examples - num_test_examples * 2

        # Build `Instance`s from `examples`.
        instances: List[Instance] = []
        for i, example in enumerate(examples):
            # Build references.
            references: List[Reference]

            # Each example has "input" and either "target_scores" or "target".
            if "target_scores" in example:
                # For "target_scores", BIG-bench compares target scores against the model's predicted probabilities.
                # It seems almost all BIG-bench Lite tasks with target scores either have a target score
                # of 0 (incorrect answer) or 1 (correct answer).
                # So, for now, `Reference`s with the highest target score are correct.
                highest_score = max(example["target_scores"].values())
                references = [
                    Reference(output, tags=[CORRECT_TAG] if score == highest_score else [])
                    for output, score in example["target_scores"].items()
                ]
            elif "target" in example:
                # All the outputs in "target" are correct e.g., {"input": "1 + 1 = ", "target": ["two","2"]}.
                # "target" can either be a list of correct values or a single correct value.
                targets: List = example["target"] if type(example["target"]) == list else [example["target"]]
                references = [Reference(output, tags=[CORRECT_TAG]) for output in targets]
            else:
                raise ValueError(f"Invalid example that doesn't have `target` or `target_scores` field: {example}")

            # Get split based on current index `i`.
            split: str
            if i < num_train_examples:
                split = TRAIN_SPLIT
            elif num_train_examples <= i < num_train_examples + num_test_examples:
                split = TEST_SPLIT
            else:
                split = VALID_SPLIT

            instances.append(Instance(example["input"], references, split=split))

        return instances
