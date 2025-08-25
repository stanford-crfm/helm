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


class ArabicMMLUScenario(Scenario):
    """ArabicMMLU

    ArabicMMLU is the first multi-task language understanding benchmark
    for Arabic language, sourced from school exams across diverse educational
    levels in different countries spanning North Africa, the Levant, and the
    Gulf regions. The data comprises 40 tasks and 14,575 multiple-choice questions
    in Modern Standard Arabic (MSA), and is carefully constructed by collaborating
    with native speakers in the region.

    - https://huggingface.co/datasets/MBZUAI/ArabicMMLU
    - https://aclanthology.org/2024.findings-acl.334/
    """

    name = "arabic_mmlu"
    description = "Arabic Massive Multitask Language Understanding"
    tags = ["knowledge", "multiple_choice"]

    OPTIONS = ["A", "B", "C", "D"]
    HF_SPLIT_TO_HELM_SPLIT = {"dev": TRAIN_SPLIT, "test": TEST_SPLIT}

    def __init__(self, subset: str):
        super().__init__()
        self.subset = subset.replace("_", " ")

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset_splits: Dict[str, datasets.Dataset] = datasets.load_dataset(
            "MBZUAI/ArabicMMLU",
            self.subset,
            revision="7aa530e2893ac420352b3f5c1a1310c010e9758b",
            cache_dir=cache_dir,
        )

        # Read all instances
        instances: List[Instance] = []
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
                            # Need to convert column to string because the references are floats
                            # for the subject "Math (Primary School)"
                            output=Output(text=str(row[column_name])),
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
