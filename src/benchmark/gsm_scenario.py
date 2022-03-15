import jsonlines
import os
from typing import List

from common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TRAIN_SPLIT, TEST_SPLIT

BASE_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"


class GSM8KScenario(Scenario):
    """Evaluate the capacity of a model to solve grade school math problems, when prompted to include reasoning"""

    name = "gsm"
    description = "Grade school math dataset with 8.5K examples (GSM8K)."
    tags = ["reasoning", "math"]

    def __init__(self, use_pilot_data=True):
        self.use_pilot_data = use_pilot_data

    def get_instances(self) -> List[Instance]:

        splits = {"train": TRAIN_SPLIT, "test": TEST_SPLIT}
        # Read all the instances
        instances = []
        for split, split_tag in splits.items():
            source_url = BASE_URL.format(split=split)
            data_path = os.path.join(self.output_path, "data")
            ensure_file_downloaded(
                source_url=source_url, target_path=data_path,
            )
            with jsonlines.open(data_path) as reader:
                for example in reader:
                    instances.append(
                        Instance(
                            input=example["question"],
                            references=[Reference(output=example["answer"], tags=[CORRECT_TAG])],
                            split=split_tag,  # Must assign split tag to instance.
                        ),
                    )
        return instances
