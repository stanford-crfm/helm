import collections
import typing
from typing import List
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, TEST_TAG, CORRECT_TAG

from datasets import load_dataset, DatasetDict


class MATHScenario(Scenario):
    """
    The MATH dataset from the paper
    "Measuring Mathematical Problem Solving With the MATH Dataset"
    by Hendrycks et al. (2021):

        https://arxiv.org/pdf/2103.03874.pdf
    """

    name = "MATH"
    description = "Mathematical Problem Solving"
    tags = ["knowledge", "reasoning"]

    types_mapping = {
        "number_theory": "Number Theory",
        "intermediate_algebra": "Intermediate Algebra",
        "algebra": "Algebra",
        "prealgebra": "Prealgebra",
        "geometry": "Geometry",
        "counting_and_probability": "Counting & Probability",
        "precalculus": "Precalculus",
    }
    levels = ["1", "2", "3", "4", "5"]

    def __init__(self, type: str, level: str):
        self.type = type
        self.level = level

    def get_instances(self) -> List[Instance]:
        dataset = {}
        data = typing.cast(DatasetDict, load_dataset("competition_math", ignore_verifications=True))

        def group_by_key(dataset_list, key):
            dataset_per_key = collections.defaultdict(list)
            for ex in dataset_list:
                dataset_per_key[ex[key]].append(ex)
            return dataset_per_key

        instances = []
        for tag, tag_name in zip([TRAIN_TAG, TEST_TAG], ["train", "test"]):
            data_split = [ex for ex in data[tag_name]]
            dataset[tag] = group_by_key(data_split, "type")
            dataset[tag] = dataset[tag][MATHScenario.types_mapping[self.type]]
            dataset[tag] = group_by_key(data_split, "level")
            dataset[tag] = dataset[tag][f"Level {self.level}"]

            for ex in dataset[tag]:
                instance = Instance(
                    input=ex["problem"], references=[Reference(output=ex["solution"], tags=[CORRECT_TAG])], tags=[tag],
                )
                instances.append(instance)

        return instances
