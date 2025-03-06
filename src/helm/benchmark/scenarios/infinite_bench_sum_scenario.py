import os
from typing import List
from datasets import load_dataset, Features, Value, Sequence, Dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.general import ensure_directory_exists


class InfiniteBenchSumScenario(Scenario):
    """InfiniteBenchSum

    InfiniteBenchbenchmark tailored for evaluating the capabilities of language models to process,
    understand, and reason over super long contexts (100k+ tokens). InfiniteBenchSum is a subset of
    InfiniteBench that requires models to generate a concise summary of the novel. The subset is referred
    to as "En.Sum" in the original paper.
    """

    name = "infinite_bench_sum"
    description = "Summarize a novel from InfiniteBench"
    tags = ["summarization"]

    def __init__(self, word_lower_bound: int = 0, word_upper_bound: int = 100e6):
        self.word_lower_bound = word_lower_bound
        self.word_upper_bound = word_upper_bound
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        # Get InfiniteBench from HuggingFace
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        # Define the features schema
        ft = Features(
            {
                "id": Value("int64"),
                "context": Value("string"),
                "input": Value("string"),
                "answer": Sequence(Value("string")),
                "options": Sequence(Value("string")),
            }
        )

        # Load the dataset with the specified features
        dataset = load_dataset("xinrongzhang2022/InfiniteBench", split="longbook_sum_eng", features=ft)

        assert isinstance(dataset, Dataset)

        dataset = dataset.map(lambda example: {"prompt": example["context"] + "\n\n" + example["input"]})
        dataset = dataset.map(lambda example: {"prompt_wc": len(example["prompt"].split())})
        dataset = dataset.filter(lambda example: self.word_lower_bound <= example["prompt_wc"] <= self.word_upper_bound)

        # Read all instances
        instances: List[Instance] = []
        for row in dataset:
            id = row["id"]
            input = Input(text=row["prompt"])
            instance = Instance(
                id=id,
                input=input,
                references=[Reference(Output(text=row["answer"][0]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances
