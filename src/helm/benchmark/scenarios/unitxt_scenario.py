from typing import List
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Output,
    Reference,
    Scenario,
    Instance,
    Input,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    VALID_SPLIT,
)


class UnitxtScenario(Scenario):
    """Integration with Unitxt: https://unitxt.rtfd.io/"""

    name = "unitxt"
    description = "Unitxt Scenarios"
    tags = ["unitxt"]

    UNITXT_SPLIT_NAME_TO_HELM_SPLIT_NAME = {
        "train": TRAIN_SPLIT,
        "test": TEST_SPLIT,
        "validation": VALID_SPLIT,
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def get_instances(self, output_path: str) -> List[Instance]:
        if len(self.kwargs) == 1 and "recipe" in self.kwargs:
            dataset_name = self.kwargs["recipe"]
        else:
            dataset_name = ",".join(f"{key}={value}" for key, value in self.kwargs.items())
        dataset = load_dataset("unitxt/data", dataset_name, trust_remote_code=True)

        instances: List[Instance] = []

        for unitxt_split_name, helm_split_name in UnitxtScenario.UNITXT_SPLIT_NAME_TO_HELM_SPLIT_NAME.items():
            dataset_split = dataset.get(unitxt_split_name)
            if dataset_split is None:
                continue
            for index, row in enumerate(dataset_split):
                references = [
                    Reference(
                        output=Output(text=reference_text),
                        tags=[CORRECT_TAG],
                    )
                    for reference_text in row["references"]
                ]
                instance = Instance(
                    id=f"{unitxt_split_name}{index}",
                    input=Input(text=row["source"]),
                    references=references,
                    split=helm_split_name,
                )
                instances.append(instance)
        return instances
