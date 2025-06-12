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


class InfiniteBenchEnQAScenario(Scenario):
    """InfiniteBench En.QA

    InfiniteBench is a benchmark tailored for evaluating the capabilities of language models to process,
    understand, and reason over long contexts (100k+ tokens). InfiniteBench En.QA is a subset of
    InfiniteBench that requires models to perform open-form question answering on questions that necessitate
    long-range dependency and reasoning, beyond simple short passage retrieval.
    """

    name = "infinite_bench_en_qa"
    description = "∞Bench En.QA is a summarization task that requires generating a concise summary of a novel. ([Zhang et al., 2024](https://arxiv.org/abs/2402.13718))"  # noqa: E501
    tags = ["question_answering"]

    def __init__(self, max_num_words: int):
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
            split="longbook_qa_eng",
            features=ft,
            revision="90f0394333616266d9fe85824ceaf505093cbaa5",
        )

        assert isinstance(dataset, Dataset)

        def count_words(text: str) -> int:
            return len(re.split(r"\s+", text.strip()))

        dataset = dataset.filter(
            lambda example: count_words(example["context"])
            + count_words(example["input"])
            + sum(count_words(option) for option in example["options"])
            <= self.max_num_words
        )

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
            )
            instances.append(instance)

        return instances
