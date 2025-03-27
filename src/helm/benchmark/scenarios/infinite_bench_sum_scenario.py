import os
import re
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
    """InfiniteBench Sum

    InfiniteBench is a benchmark tailored for evaluating the capabilities of language models to process,
    understand, and reason over super long contexts (100k+ tokens). InfiniteBench Sum is a subset of
    InfiniteBench that requires models to generate a concise summary of the novel. The subset is referred
    to as "En.Sum" in the original paper.
    """

    name = "infinite_bench_sum"
    description = "Summarize a novel from InfiniteBench"
    tags = ["summarization"]

    def __init__(self, min_num_words: int, max_num_words: int):
        self.min_num_words = min_num_words
        self.max_num_words = max_num_words
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
        dataset = load_dataset(
            "xinrongzhang2022/InfiniteBench",
            split="longbook_sum_eng",
            features=ft,
            revision="90f0394333616266d9fe85824ceaf505093cbaa5",
        )

        assert isinstance(dataset, Dataset)

        def count_words(text: str) -> int:
            return len(re.split(r"\s+", text.strip()))

        dataset = dataset.map(
            lambda example: {"prompt_wc": count_words(example["context"]) + count_words(example["input"])}
        ).filter(lambda example: self.min_num_words <= example["prompt_wc"] <= self.max_num_words)

        # Read all instances
        instances: List[Instance] = []
        for row in dataset:
            id = row["id"]
            input = Input(text=row["context"] + "\n\n" + row["input"])
            instance = Instance(
                id=id,
                input=input,
                references=[Reference(Output(text=row["answer"][0]), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
                extra_data={"word_count": row["prompt_wc"]},
            )
            instances.append(instance)

        return instances
