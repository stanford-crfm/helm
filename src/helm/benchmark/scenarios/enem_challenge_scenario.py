import re
from typing import List, Any, Dict
from datasets import load_dataset

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


class ENEMChallengeScenario(Scenario):
    """
    This scenario is based on the National High School Exams that were applied throughout the years
    of 2009 and 2023.

    The examples are questions about all types of intelectual fields such as mathemathics and grammar.
    """

    name = "enem_challenge"
    description = "ENEM Challenge dataset"
    tags = ["knowledge", "multiple_choice", "pt-br"]

    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data and read all the dialogues
        dataset: Any
        # Read all the instances
        instances: List[Instance] = []

        dataset = load_dataset("eduagarcia/enem_challenge")
        for example in dataset["train"]:
            question = example["question"]
            choices = example["choices"]
            answer = example["answerKey"]
            # Skipping every canceled question!
            if answer == "ANULADO":
                continue
            answers_dict = dict(zip(choices["label"], choices["text"]))
            correct_answer = answers_dict[answer]

            def answer_to_reference(answer: str) -> Reference:
                return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])

            instance = Instance(
                input=Input(text=question), split=TEST_SPLIT, references=list(map(answer_to_reference, choices["text"]))
            )
            instances.append(instance)
        return instances
