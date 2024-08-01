import re
from typing import List, Any, Dict
from datasets import load_dataset

from .scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class SimpleSafetyTestsScenario(Scenario):
    """
    The data contains a diverse set of prompts covering 70 hypothetical decision scenarios, ranging
    from approving a loan to providing press credentials. Each prompt instructs the model to make a
    binary decision (yes/no) about a particular person described in the prompt.
    https://huggingface.co/datasets/Anthropic/discrim-eval
    """

    name = "SimpleSafetyTests"
    description = "SimpleSafetyTests dataset"
    tags = ["safety"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # read explicit data
        dataset = load_dataset("Bertievidgen/SimpleSafetyTests", split="test")

        # Read all the instances
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["prompt"])
            references = [
                Reference(output=Output(text=row[column_name]), tags=[]) for column_name in ["category", "harm_area"]
            ]
            instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)
        return instances
