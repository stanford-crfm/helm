import random
import os
import json
import datasets
from pathlib import Path
from typing import List, Dict

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    Input,
    Output,
)

PROMPT_SETTINGS_URL = "https://raw.githubusercontent.com/HazyResearch/legalbench/main/helm_prompt_settings.jsonl"

SUBSETS = [
    "abercrombie",
    "corporate_lobbying",
    "international_citizenship_questions",
    "function_of_decision_section",
    "proa",
]


def get_legalbench_prompt_settings(subset: str, cache_dir: str):
    """
    Loads prompt construction settings for all subsets.
    """
    assert subset in SUBSETS, "Unknown subset: {}".format(subset)

    prompt_construction_settings_path = os.path.join(cache_dir, "prompt_construction_settings.json")
    ensure_directory_exists(cache_dir)
    ensure_file_downloaded(
        source_url=PROMPT_SETTINGS_URL,
        target_path=prompt_construction_settings_path,
    )
    with open(prompt_construction_settings_path, "r") as f:
        field_ordering, instructions, label_keys, output_nouns, _ = map(json.loads, f.read().strip().split("\n"))
    return (
        field_ordering[subset],
        instructions[subset],
        label_keys[subset],
        output_nouns[subset],
    )


def get_legalbench_instructions(subset: str, cache_dir: str):
    return get_legalbench_prompt_settings(subset, cache_dir)[1]


def get_legalbench_output_nouns(subset: str, cache_dir: str):
    return get_legalbench_prompt_settings(subset, cache_dir)[3]


class LegalBenchScenario(Scenario):
    """
    LegalBench is benchmark containing different legal reasoning tasks. We use a subset of the tasks, selected
    to represent different legal reasoning patterns.

    LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models
    https://arxiv.org/abs/2308.11462

    Official website for LegalBench:
    http://hazyresearch.stanford.edu/legalbench/

    Dataset summary:
    https://huggingface.co/datasets/nguha/legalbench

    Prompts are adapted from:
    https://github.com/HazyResearch/legalbench/

    Subsets:

    - abercrombie
    - corporate_lobbying
    - international_citizenship_questions
    - function_of_decision_section
    - proa
    """

    name = "legalbench"
    description = "LegalBench"
    tags = ["text_classification", "robustness"]

    def __init__(self, subset: str, random_seed=42):
        super().__init__()
        assert subset in SUBSETS, "Unknown subset: {}".format(subset)
        self.subset = subset
        self.random_seed = random_seed

    def load_prompt_construction_settings(self, output_path: str):
        # Load from prompt construction settings
        cache_dir = str(Path(output_path) / "data")
        return get_legalbench_prompt_settings(self.subset, cache_dir)

    def get_instances(self, output_path: str) -> List[Instance]:
        fields, _, label_key, _ = self.load_prompt_construction_settings(output_path)
        cache_dir = str(Path(output_path) / "data")

        # Download data from Huggingface. LegalBench provides splits for samples to
        # be used for prompt construction and for testing.
        train_dataset = datasets.load_dataset(
            "nguha/legalbench",
            self.subset,
            trust_remote_code=True,
            cache_dir=cache_dir,
            split="train",
            revision="e042ea68c19df12b737fe768572f22ead61e8e37",
        )
        test_dataset = datasets.load_dataset(
            "nguha/legalbench",
            self.subset,
            trust_remote_code=True,
            cache_dir=cache_dir,
            split="test",
            revision="e042ea68c19df12b737fe768572f22ead61e8e37",
        )
        assert isinstance(train_dataset, datasets.Dataset)
        assert isinstance(test_dataset, datasets.Dataset)

        dataset_splits: Dict[str, datasets.Dataset] = {
            TRAIN_SPLIT: train_dataset,
            TEST_SPLIT: test_dataset,
        }

        # Read all instances
        random.seed(self.random_seed)
        instances: List[Instance] = []
        for split, subset in dataset_splits.items():
            for x in subset:
                assert fields is not None, "Field ordering not loaded"
                prompt: str = "\n".join([f"{field[0]}: {x[field[1]]}" for field in fields])
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=x[label_key]), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
