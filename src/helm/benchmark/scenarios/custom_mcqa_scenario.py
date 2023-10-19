import csv
import os
from typing import Dict, Optional, List

from helm.common.hierarchical_logger import hlog

# from .scenario import Scenario, Instance, Reference, CORRECT_TAG, Input, Output
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, CORRECT_TAG, Input, Output


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
        self._ensure_file_exist_and_is_well_formatted(eval_path)
        self.eval_path: str = eval_path

        # Train path (optional but should be provided if num_train_instances > 0)
        self.train_path: Optional[str] = None
        if num_train_instances > 0:
            if train_path is None:
                raise ValueError("train_path must be provided if num_train_instances > 0")
            self._ensure_file_exist_and_is_well_formatted(train_path)
            self.train_path = train_path

        # Set the name, description, and tags
        CustomMCQAScenario.name = name if name is not None else CustomMCQAScenario.name
        CustomMCQAScenario.description = description if description is not None else CustomMCQAScenario.description
        if tags is not None:
            CustomMCQAScenario.tags.extend(tags)

    def _ensure_file_exist_and_is_well_formatted(self, path: str) -> None:
        """Check if the file exists and is well formatted.

        Raises:
            ValueError: If the file is not well formatted.
        """

        # Check if the file exists
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist.")

        # Check if the file is well formatted
        row_index: int = 0
        try:
            with open(path) as f:
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    # Example: ["What color is the sky?", "red", "blue", "green", "B"]
                    # CHecks that the last element is a single capital letter
                    if len(row[-1]) != 1 and not row[-1].isalpha() or not row[-1].isupper():
                        raise ValueError(
                            "The last element of each row must be a single capital letter "
                            "corresponding to the correct answer."
                        )
                    if len(row) < 3:
                        raise ValueError("There must be at least 3 elements per row.")
                    elif len(row) == 3:
                        hlog(
                            f"WARNING: There is only one answer for the question {row[0]} in row {row_index}. "
                            f"Please make sure that this is what you want."
                        )
                    row_index += 1
        except Exception as e:
            raise ValueError(f"File {path} is not well formatted. Found error in row {row_index}. Error: {e}") from e

    def process_csv(self, csv_path: str, split: Optional[str] = None) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Reading {csv_path}")
        with open(csv_path) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                # Example: ["What color is the sky?", "red", "blue", "green", "B"]
                question, answers, correct_choice = row[0], row[1:-1], row[-1]
                answers_dict = dict(zip(["A", "B", "C", "D", "E"], answers))
                correct_answer: str = answers_dict[correct_choice]

                def answer_to_reference(answer: str) -> Reference:
                    return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

                instance = Instance(
                    input=Input(text=question),
                    references=list(map(answer_to_reference, answers)),
                    split=split,
                )
                instances.append(instance)
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
            instances.extend(self.process_csv(path, split=split))
        return instances
