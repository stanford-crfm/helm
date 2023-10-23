import os
from typing import Dict, Optional, List

from helm.common.codec import from_jsonl
from .scenario import Scenario, Instance, TRAIN_SPLIT, VALID_SPLIT


class CustomMCQAScenario(Scenario):
    """
    A Generic Multiple Choice Question Answering Scenario that a user can use to
    benchmark a model on their own data.
    Code is adapted from the MMLU scenario.

    We prompt models using the following format

        <input>                  # train
        A. <reference>
        B. <reference>
        C. <reference>
        D. <reference>
        Answer: <A/B/C/D>

        x N (N-shot)

        <input>                  # test
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer:

    For example (from mmlu:anatomy), we have:

        The pleura
        A. have no sensory innervation.
        B. are separated by a 2 mm space.
        C. extend into the neck.
        D. are composed of respiratory epithelium.
        Answer: C

        Which of the following terms describes the body's ability to maintain its normal state?
        A. Anabolism
        B. Catabolism
        C. Tolerance
        D. Homeostasis
        Answer:

    Target: D
    """

    name = ""  # Will be set by the user in constructor
    description = ""  # Will be set by the user in constructor
    tags = ["multiple_choice"]  # Will be concatenated with the user's tags in constructor

    def __init__(
        self,
        eval_path: str,
        name: str,
        description: str,
        train_path: Optional[str],
        num_train_instances: int = 0,
        tags: Optional[List[str]] = None,
    ):
        super().__init__()

        # Eval path (should always be provided)
        if not os.path.exists(eval_path):
            raise ValueError(f"eval_path {eval_path} does not exist")
        self.eval_path: str = eval_path

        # Train path (optional but should be provided if num_train_instances > 0)
        self.train_path: Optional[str] = None
        if num_train_instances > 0:
            if train_path is None:
                raise ValueError("train_path must be provided if num_train_instances > 0")
            if not os.path.exists(train_path):
                raise ValueError(f"train_path {train_path} does not exist")
            self.train_path = train_path

        # Set the name, description, and tags
        CustomMCQAScenario.name = name if name is not None else CustomMCQAScenario.name
        CustomMCQAScenario.description = description if description is not None else CustomMCQAScenario.description
        if tags is not None:
            CustomMCQAScenario.tags.extend(tags)

    def process_jsonl(self, jsonl_path: str, split: Optional[str] = None) -> List[Instance]:
        instances: List[Instance] = []
        with open(jsonl_path, mode="r") as reader:
            instances = from_jsonl(reader.read(), Instance)

        # Override the split if provided
        if split is not None:
            for i, instance in enumerate(instances):
                # Since instance is a frozen dataclass, we need to create a new instance
                # with the updated split (only if split is different from the current split)
                if instance.split != split:
                    instances[i] = Instance(
                        input=instance.input,
                        references=instance.references,
                        split=split,
                    )

        return instances

    def get_instances(self) -> List[Instance]:
        # Read all the instances
        instances: List[Instance] = []
        split_to_path: Dict[str, Optional[str]] = {
            VALID_SPLIT: self.eval_path,
            TRAIN_SPLIT: self.train_path,
        }
        for split, path in split_to_path.items():
            if path is None:
                continue
            instances.extend(self.process_jsonl(path, split=split))
        return instances
