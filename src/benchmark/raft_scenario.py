import random
import os
import json
import tempfile
import datasets
from pathlib import Path
from common.general import ensure_file_downloaded
from typing import List, Dict
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, TEST_TAG, CORRECT_TAG

# TODO: replace with permalink
PROMPT_SETTINGS_URL = "https://www.dropbox.com/s/a5cyevryzw8rt4f/prompt_construction_settings.json?dl=0"


def get_raft_instructions(subset: str):
    # Load prompt construction settings
    tmp_dir = tempfile.gettempdir()
    prompt_construction_settings_path = os.path.join(tmp_dir, "prompt_construction_settings.json")
    ensure_file_downloaded(
        source_url=PROMPT_SETTINGS_URL, target_path=prompt_construction_settings_path,
    )
    with open(prompt_construction_settings_path, "r") as f:
        field_ordering, instructions = map(json.loads, f.read().strip().split("\n"))

    assert subset in instructions, "Unknown subset: {}".format(subset)

    return instructions[subset]


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
        ade_corpus_v2
        banking_77
        neurips_impact_statement_risks
        one_stop_english
        overruling
        semiconductor_org_types
        systematic_review_inclusion
        tai_safety_research
        terms_of_service
        tweet_eval_hate
        twitter_complaints
    """

    name = "raft"
    description = "Real-world Annotated Few-shot Tasks (RAFT)"
    tags = ["text_classification", "robustness"]

    def __init__(self, subset: str, random_seed=42):
        self.subset = subset
        self.random_seed = random_seed
        self.instructions = get_raft_instructions(subset)
        self.fields = None
        random.seed(random_seed)

    def load_prompt_construction_settings(self):
        # Skip if already loaded
        if self.fields:
            return
        # Download prompt construction settings
        prompt_construction_settings_path = str(Path(self.output_path) / "data" / "prompt_construction_settings.json")
        ensure_file_downloaded(
            source_url=PROMPT_SETTINGS_URL, target_path=prompt_construction_settings_path,
        )
        with open(prompt_construction_settings_path, "r") as f:
            FIELD_ORDERING, _ = map(json.loads, f.read().strip().split("\n"))
        self.fields = FIELD_ORDERING[self.subset]

    def get_instances(self) -> List[Instance]:
        self.load_prompt_construction_settings()
        cache_dir = str(Path(self.output_path) / "data")
        # Download raw data
        # TODO: Only using labeled instances now. Check if we can get the hidden test set labels.
        all_usable_dataset = datasets.load_dataset("ought/raft", self.subset, cache_dir=cache_dir, split="train")
        assert isinstance(all_usable_dataset, datasets.Dataset)
        dataset = all_usable_dataset.train_test_split(test_size=0.2, seed=self.random_seed)
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        class_label_to_string = train_dataset.features["Label"].int2str

        dataset_splits: Dict[str, datasets.Dataset] = {
            TRAIN_TAG: train_dataset,
            TEST_TAG: test_dataset,
        }

        # # Read all instances
        instances: List[Instance] = []
        for tag, subset in dataset_splits.items():
            for x in subset:
                assert self.fields is not None, "Field ordering not loaded"
                prompt = "\n".join([f"{field}: {x[field]}" for field in self.fields])
                instance = Instance(
                    input=prompt,
                    references=[Reference(output=class_label_to_string(x["Label"]), tags=[CORRECT_TAG])],
                    tags=[tag],
                )
                instances.append(instance)

        return instances
