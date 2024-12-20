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

PROMPT_SETTINGS_URL = "https://www.dropbox.com/s/a5cyevryzw8rt4f/prompt_construction_settings.json?dl=0"

SUBSETS = [
    "ade_corpus_v2",
    "banking_77",
    "neurips_impact_statement_risks",
    "one_stop_english",
    "overruling",
    "semiconductor_org_types",
    "systematic_review_inclusion",
    "tai_safety_research",
    "terms_of_service",
    "tweet_eval_hate",
    "twitter_complaints",
]


def get_raft_prompt_settings(subset: str, cache_dir: str):
    assert subset in SUBSETS, "Unknown subset: {}".format(subset)

    prompt_construction_settings_path = os.path.join(cache_dir, "prompt_construction_settings.json")
    ensure_directory_exists(cache_dir)
    ensure_file_downloaded(
        source_url=PROMPT_SETTINGS_URL,
        target_path=prompt_construction_settings_path,
    )
    with open(prompt_construction_settings_path, "r") as f:
        field_ordering, instructions = map(json.loads, f.read().strip().split("\n"))

    return field_ordering[subset], instructions[subset]


def get_raft_instructions(subset: str, cache_dir: str) -> str:
    return get_raft_prompt_settings(subset, cache_dir)[1]


class RAFTScenario(Scenario):
    """
    RAFT: A Real-World Few-Shot Text Classification Benchmark
    https://arxiv.org/abs/2109.14076

    Official website for RAFT dataset:
    https://raft.elicit.org/

    Dataset summary:
    https://huggingface.co/datasets/ought/raft/blob/main/README.md

    Prompts are adapted from:
    https://github.com/oughtinc/raft-baselines/tree/master/example_prompts

    Subsets:

    - ade_corpus_v2
    - banking_77
    - neurips_impact_statement_risks
    - one_stop_english
    - overruling
    - semiconductor_org_types
    - systematic_review_inclusion
    - tai_safety_research
    - terms_of_service
    - tweet_eval_hate
    - twitter_complaints

    Prompt format

        Sentence: <sentence>
        Label: <label>

    Examples from ADE corpus (adverse drug effect):

        Sentence: No regional side effects were noted.
        Label: not ADE-related
    """

    name = "raft"
    description = "Real-world Annotated Few-shot Tasks (RAFT)"
    tags = ["text_classification", "robustness"]

    def __init__(self, subset: str, random_seed=42):
        super().__init__()
        assert subset in SUBSETS, "Unknown subset: {}".format(subset)
        self.subset = subset
        self.random_seed = random_seed

    def load_prompt_construction_settings(self, output_path: str):
        # Load from prompt construction settings
        cache_dir = str(Path(output_path) / "data")
        return get_raft_prompt_settings(self.subset, cache_dir)

    def get_instances(self, output_path: str) -> List[Instance]:
        fields, _ = self.load_prompt_construction_settings(output_path)
        cache_dir = str(Path(output_path) / "data")
        # Download raw data
        # Note: Only using public labeled instances now. Check if we can get the hidden test set labels.
        all_usable_dataset = datasets.load_dataset(
            "ought/raft",
            self.subset,
            cache_dir=cache_dir,
            split="train",
            revision="9ee50172ea9afda2f1033c6f1b986e568b862fb3",
        )
        assert isinstance(all_usable_dataset, datasets.Dataset)
        dataset = all_usable_dataset.train_test_split(test_size=0.8, seed=self.random_seed)
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        class_label_to_string = train_dataset.features["Label"].int2str

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
                prompt: str = "\n".join([f"{field}: {x[field]}" for field in fields])
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=class_label_to_string(x["Label"])), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
