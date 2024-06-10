import json
import os
from typing import List

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


class CIMCQAScenario(Scenario):
    """CIMCQA is a multiple-choice question answering (MCQA) dataset designed to
    study concept inventories in CS Education.

    This is used by a pre-publication paper.

    NOTE: This code is for archival purposes only. The scenario cannot be run because it requires
    private data. Please contact the paper authors for more information."""

    DATASET_DOWNLOAD_URL: str = "https://drive.google.com/uc?export=download&id=1siYjhDiasI5FIiS0ckLbo40UnOj8EU2h"

    name = "ci_mcqa"
    description = (
        "CIMCQA is a multiple-choice question answering (MCQA) dataset designed to"
        "study concept inventories in CS Education."
    )
    tags = ["question_answering"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join("restricted", "bdsi_multiple_answers_removed.json")
        assert os.path.exists(data_path)

        with open(data_path, "r", encoding="utf8") as f:
            data = json.load(f)

        # Data is a list of dictionaries now, each one a question and its associated answers and metadata.
        instances: List[Instance] = list()

        # UNCOMMENT BELOW FOR FEW-SHOT RUN
        training_data_path: str = os.path.join("restricted", "mock_bdsi_multiple_answers_removed.json")
        assert os.path.exists(training_data_path)

        with open(training_data_path, "r", encoding="utf8") as f:
            training_data = json.load(f)
        for question in training_data:
            question_text = question["question"]
            references = list()
            for index, answer in enumerate(question["options"]):
                reference_answer = Output(text=answer)
                # Correct option offset by 1 due to zero-indexing
                tag = [CORRECT_TAG] if index == question["correct_option"] - 1 else []
                references.append(Reference(reference_answer, tags=tag))
            instance = Instance(
                input=Input(text=question_text),
                references=references,
                split=TRAIN_SPLIT,
            )
            instances.append(instance)

        for question in data:
            question_text = question["question"]
            references = list()
            for index, answer in enumerate(question["options"]):
                reference_answer = Output(text=answer)
                # Correct option offset by 1 due to zero-indexing
                tag = [CORRECT_TAG] if index == question["correct_option"] - 1 else []
                references.append(Reference(reference_answer, tags=tag))
            instance = Instance(
                input=Input(text=question_text),
                references=references,
                split=TEST_SPLIT,  # Just doing zero shot to start
            )
            instances.append(instance)
        return instances
