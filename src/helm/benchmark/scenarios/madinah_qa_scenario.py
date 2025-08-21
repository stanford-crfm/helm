import os
from typing import Dict, List

import datasets

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    TRAIN_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class MadinahQAScenario(Scenario):
    """MadinahQA Scenario"""

    name = "madinah_qa"
    description = "Arabic language competency benchmark"
    tags = ["language", "multiple_choice"]

    OPTIONS = ["A", "B", "C", "D"]
    HF_SPLIT_TO_HELM_SPLIT = {"dev": TRAIN_SPLIT, "test": TEST_SPLIT}
    SUBSETS = ["Arabic Language (General)", "Arabic Language (Grammar)"]

    def __init__(self, subset: str):
        super().__init__()
        subset = subset.replace("_", " ")
        if subset not in self.SUBSETS:
            raise Exception(f"Unknown subset: {subset}")
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        instances: List[Instance] = []
        dataset_splits: Dict[str, datasets.Dataset] = datasets.load_dataset(
            "MBZUAI/MadinahQA",
            self.subset,
            revision="62e7c86ac5c07245a5a952722691d77ddb41f695",
            cache_dir=cache_dir,
        )

        # Read all instances
        for split_name, dataset in dataset_splits.items():
            assert isinstance(dataset, datasets.Dataset)
            for row_index, row in enumerate(dataset):
                input = Input(text=row["Question"])
                references: List[Reference] = []
                correct_option_index = ord(row["Answer Key"]) - ord("A") + 1
                for option_index in range(1, 6):
                    column_name = f"Option {option_index}"
                    if not row[column_name]:
                        continue
                    references.append(
                        Reference(
                            output=Output(text=row[column_name]),
                            tags=[CORRECT_TAG] if option_index == correct_option_index else [],
                        )
                    )
                instance = Instance(
                    id=f"id{row_index}",
                    input=input,
                    references=references,
                    split=self.HF_SPLIT_TO_HELM_SPLIT[split_name],
                )
                instances.append(instance)

        return instances
