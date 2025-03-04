from typing import List, Any
from pathlib import Path
from datasets import load_dataset

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Output,
)


class OABExamsScenario(Scenario):
    """
    The OAB Exam is a mandatory test for anyone who wants to practice law in Brazil. The exam is composed for
    an objective test with 80 multiple-choice questions covering all areas of Law and a written phase focused
    on a specific legal area (e.g., Civil, Criminal, Labor Law), where candidates must draft a legal document
    and answer four essay questions.

    This dataset is composed by the exams that occured between 2010 and 2018.

    The dataset can be found in this link: https://huggingface.co/datasets/eduagarcia/oab_exams
    """

    name = "oab_exams"
    description = "OAB exams dataset"
    tags = ["knowledge", "multiple_choice", "pt-br"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data and read all the dialogues
        dataset: Any
        # Read all the instances
        instances: List[Instance] = []
        cache_dir = str(Path(output_path) / "data")

        dataset = load_dataset("eduagarcia/oab_exams", cache_dir=cache_dir)
        for example in dataset["train"]:
            question = example["question"]
            choices = example["choices"]
            answer = example["answerKey"]
            # Skipping every canceled question!
            if example["nullified"]:
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
