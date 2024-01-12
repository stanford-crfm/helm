from typing import List

from datasets import load_dataset

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class I2PScenario(Scenario):
    """
    The Inappropriate Image Prompts (I2P) benchmark contains real user prompts for generative text-to-image prompts
    that are likely to produce inappropriate images. The benchmark contains 4703 unique prompts. Each prompt belongs
    to one of seven categories:

    1. hate
    2. harassment
    3. violence
    4. self-harm
    5. sexual content => sexual
    6. shocking images => shocking
    7. illegal activity => illegal

    Setting parameter `category` to "all", returns instances with all the prompts.

    The authors generated 10 images for each prompt in the test bed and compared the probability of generating
    inappropriate images. We use the hard prompts that are not considered toxic by PerspectiveAPI.

    Paper: https://arxiv.org/abs/2211.05105
    Website: https://huggingface.co/datasets/AIML-TUDA/i2p
    """

    ALL_CATEGORY: str = "all"

    name = "i2p"
    description = (
        "Inappropriate Image Prompts (I2P) benchmark contains real user prompts for generative "
        "text-to-image prompts that are likely to produce inappropriate images "
        "([paper](https://arxiv.org/abs/2211.05105))."
    )
    tags = ["text-to-image", "harms", "toxicity"]

    def __init__(self, category: str):
        super().__init__()
        self.category: str = category

    def get_instances(self, _) -> List[Instance]:
        return [
            # There are sometimes random spaces at the beginning and end of the prompts
            Instance(Input(text=row["prompt"].strip()), references=[], split=TEST_SPLIT)
            # The dataset does not have separate train/test splits
            for row in load_dataset("AIML-TUDA/i2p", split="train")
            if row["prompt"]
            # Use the "hard" prompts that are not considered toxic by PerspectiveAPI.
            # The "hard" prompts are more likely to generate toxic images.
            and row["hard"] == 1
            and row["prompt_toxicity"] < 0.5
            and (self.category in row["categories"] or self.category == self.ALL_CATEGORY)
        ]
