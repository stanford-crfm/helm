import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Scenario,
    Instance,
    Reference,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists


class Banking77Scenario(Scenario):
    """BANKING77

    BANKING77 is an intent classification scenario using a very fine-grained
    set of intents in a banking domain. It comprises 13,083 customer service
    queries labeled with 77 intents.

    Paper: https://arxiv.org/abs/2003.04807"""

    name = "banking77"
    description = (
        "BANKING77 is an intent classification scenario using a very fine-grained "
        "set of intents in a banking domain. It comprises 13,083 customer service "
        "queries labeled with 77 intents."
    )
    tags = ["finance", "classification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        # TODO: Switch this to the production dataset when available.
        dataset = datasets.load_dataset(
            "PolyAI/banking77",
            cache_dir=cache_dir,
            revision="90d4e2ee5521c04fc1488f065b8b083658768c57",
            trust_remote_code=True,
        )

        instances: List[Instance] = []
        for split_name in [TRAIN_SPLIT, TEST_SPLIT]:
            dataset_split = dataset[split_name]
            label_feature = dataset_split.features["label"]
            for row in dataset_split:
                input = Input(text=row["text"])
                references = [Reference(output=Output(text=label_feature.int2str(row["label"])), tags=[CORRECT_TAG])]
                instance = Instance(input=input, references=references, split=split_name)
                instances.append(instance)
        return instances
