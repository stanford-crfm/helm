from typing import List
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, CORRECT_TAG, TEST_SPLIT, Reference, Output


class XSTestScenario(Scenario):
    """
    XSTest, a simple standardized evaluation for safety refusals.
    """

    name = "XSTest"
    description = "XSTest dataset - safety benchmark"
    tags = ["safety"]

    def __init__(self, subject: str = "all"):
        super().__init__()
        self.subset = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        # read explicit data
        dataset = load_dataset("walledai/XSTest", split="test", revision="f1d713187c61b6ae64e602d74f0b3d812cc2e8e8")

        # Read all the instances
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["prompt"])
            references = []
            for column_name in ["focus", "type", "note"]:
                if row[column_name]:
                    references += [Reference(output=Output(text=row[column_name]), tags=[])]
            references += [Reference(output=Output(text=row["label"]), tags=[CORRECT_TAG])]
            instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)
        return instances
