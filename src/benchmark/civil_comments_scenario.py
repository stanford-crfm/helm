import json
import os
import random
import datasets
from typing import List
from pathlib import Path
from tqdm import tqdm

from common.general import ensure_file_downloaded
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    CORRECT_TAG,
    TEST_SPLIT,
    DEFAULT_TEST_SIZE,
)

bool_to_response = {
    False: "False",
    True: "True",
}


class CivilCommentsScenario(Scenario):
    """
    The Civil Comments dataset is from the following Hugging Face dataset:

        https://huggingface.co/datasets/civil_comments
        
    which, in turn, is from the following Kaggle contest:
    
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
    
    Example prompts (here, ellipses are added purely for visual aid):
    
    PROMPT                                          TOXICITY    SEVERE_TOXICITY
    This is malfeasance by the Administrator        0           0
    and the Board. They are wasting our money!
    OBSCENE         THREAT      INSULT      IDENTITY_ATTACK     SEXUAL_EXPLICIT
    0               0           0           0                   0
    
    
    """

    name = "civil_comments"
    description = "A large-scale dataset that consists of 1804874 sentences from the Civil Comments platform, a commenting plugin for independent news sites."
    tags = ["toxicity", "bias"]

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        random.seed(random_seed)

    def get_instances(self) -> List[Instance]:
        cache_dir = str(Path(self.output_path) / "data")

        # Download raw data
        # TODO: Only using labeled instances now.
        all_usable_dataset = datasets.load_dataset("civil_comments", cache_dir=cache_dir, split="train")
        assert isinstance(all_usable_dataset, datasets.Dataset)
        dataset = all_usable_dataset.train_test_split(test_size=0.2, seed=self.random_seed)
        train_dataset, test_dataset = dataset["train"], dataset["test"]

        dataset_splits: Dict[str, datasets.Dataset] = {
            TRAIN_SPLIT: train_dataset,
            TEST_SPLIT: test_dataset,
        }

        # Read all instances
        instances: List[Instance] = []
        # pudb.set_trace()
        for split, subset in dataset_splits.items():
            for x in tqdm(subset):
                # print(f"x is: {x}")
                prompt = x["text"]
                instance = Instance(
                    input=prompt,
                    references=[Reference(output=bool_to_response[x["toxicity"] >= 0.5], tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances
