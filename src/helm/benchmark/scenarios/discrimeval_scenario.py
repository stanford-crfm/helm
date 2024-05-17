import re
from typing import List, Any, Dict
from datasets import load_dataset

from .scenario import Scenario, Instance, Input, TRAIN_SPLIT, TEST_SPLIT


class DiscrimEvalScenario(Scenario):
    """
    The data contains a diverse set of prompts covering 70 hypothetical decision scenarios, ranging
    from approving a loan to providing press credentials. Each prompt instructs the model to make a
    binary decision (yes/no) about a particular person described in the prompt.

    https://huggingface.co/datasets/Anthropic/discrim-eval
    """

    name = "discrimeval"
    description = "DiscrimEval dataset"
    tags = ["safety", "bias"]

    def __init__(self, subject: str = "all"):
        super().__init__()
        self.subset = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        # read explicit data
        dataset = load_dataset("Anthropic/discrim-eval", "explicit")

        # Read all the instances
        instances: List[Instance] = []
        for example in dataset:
            instance = Instance(
                input=Input(text=example["filled_template"]),
                references=[],
            )
            instances.append(instance)
        return instances
