import json
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
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


class MoonshotScenario(Scenario):
    name = "moonshot"
    description = "Moonshot scenario"
    tags = ["reasoning", "math"]

    _BASE_URL = "https://raw.githubusercontent.com/aiverify-foundation/moonshot-data/main/datasets/"

    def __init__(self, dataset_ids: List[str]):
        self.dataset_ids = dataset_ids
        super().__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for dataset_id in self.dataset_ids:
            file_name = f"{dataset_id}.json"
            source_url: str = f"{MoonshotScenario._BASE_URL}/{file_name}"
            file_path: str = os.path.join(output_path, file_name)

            ensure_file_downloaded(source_url=source_url, target_path=file_path)

            with open(file_path, "r") as f:
                dataset = json.load(f)
            for example in dataset["examples"]:
                input = Input(text=example["input"])
                references = [
                    Reference(Output(text=example["target"]), tags=[CORRECT_TAG]),
                ]
                instance = Instance(input=input, references=references, split=TEST_SPLIT)
                instances.append(instance)

        return instances
