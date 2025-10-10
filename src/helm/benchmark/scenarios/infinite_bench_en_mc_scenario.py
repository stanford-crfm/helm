import os
import re
from typing import List

from datasets import load_dataset, Features, Value, Sequence, Dataset

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
    ScenarioMetadata,
)
from helm.common.general import ensure_directory_exists


class InfiniteBenchEnMCScenario(Scenario):
    """InfiniteBench En.MC

    InfiniteBench is a benchmark tailored for evaluating the capabilities of language models to process,
    understand, and reason over long contexts (100k+ tokens). InfiniteBench En.MC is a subset of
    InfiniteBench that requires models to perform multiple-choice question answering on questions that necessitate
    long-range dependency and reasoning, beyond simple short passage retrieval.
    """

    name = "infinite_bench_en_mc"
    description = "∞Bench En.MC is a multiple-choice question answering task that necessitates long-range dependency and reasoning. ([Zhang et al., 2024](https://arxiv.org/abs/2402.13718))"  # noqa: E501
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
            split="longbook_choice_eng",
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
            assert len(row["answer"]) == 1
            id = row["id"]
            input = Input(text=row["context"] + "\n\n" + row["input"])
            references = [
                Reference(Output(text=option), tags=[CORRECT_TAG] if option == row["answer"][0] else [])
                for option in row["options"]
            ]
            instance = Instance(
                id=id,
                input=input,
                references=references,
                split=TEST_SPLIT,
            )
            instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="infinite_bench_en_mc",
            display_name="∞Bench En.MC",
            description="∞Bench En.MC is a multiple-choice question answering task that requires "
            "locating and processing information within a novel, performing reasoning "
            "through aggregation or filtering to derive answers. ([Zhang et al., "
            "2024](https://arxiv.org/abs/2402.13718))",
            taxonomy=TaxonomyInfo(
                task="multiple-choice question answering",
                what="Novels",
                when="Before 2024",
                who="Novel authors",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
