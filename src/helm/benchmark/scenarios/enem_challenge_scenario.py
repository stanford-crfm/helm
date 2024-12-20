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


class ENEMChallengeScenario(Scenario):
    """
    The Exame Nacional do Ensino Médio (ENEM) is an advanced High-School level exam widely applied
    every year by the Brazilian government to students that wish to undertake a University degree.

    The questions are about all types of intelectual fields and they are divided into four groups
    that are named as: Humanities, Languages, Sciences and Mathematics.

    This scenario is based on the exams that were applied throughout the years of 2009 and 2023.

    The dataset can be found in this link: https://huggingface.co/datasets/eduagarcia/enem_challenge
    """

    name = "enem_challenge"
    description = "ENEM Challenge dataset"
    tags = ["knowledge", "multiple_choice", "pt-br"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the raw data and read all the dialogues
        dataset: Any
        # Read all the instances
        instances: List[Instance] = []
        cache_dir = str(Path(output_path) / "data")

        dataset = load_dataset("eduagarcia/enem_challenge", cache_dir=cache_dir)
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
