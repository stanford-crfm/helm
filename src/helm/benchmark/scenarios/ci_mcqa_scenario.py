import json
import os
from typing import Dict, List

from helm.common.general import ensure_file_downloaded
from .scenario import Scenario, Instance, Reference, CORRECT_TAG, TRAIN_SPLIT, VALID_SPLIT, Input, Output


class CIMCQAScenario(Scenario):
    """
    Add comment describing data set and giving example from
    the data set once you have it ready.
    """

    # From https://github.com/MedMCQA/MedMCQA#data-fields, there are four possible answer choices
    # where "cop" corresponds to the index of the correct option.

    # How to account for cases with more than 4 options/variable options?
    # ANSWER_OPTION_TO_INDEX: Dict[str, int] = {"opa": 1, "opb": 2, "opc": 3, "opd": 4}

    DATASET_DOWNLOAD_URL: str = (
        "https://drive.google.com/uc?export=download&id=1siYjhDiasI5FIiS0ckLbo40UnOj8EU2h"
    )

    name = "ci_mcqa"
    description = (
        "CIMCQA is a multiple-choice question answering (MCQA) dataset designed to"
        "study concept inventories in CS Education."
    )
    tags = ["question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join("restricted", "scs1.json")
        assert os.path.exists(data_path)

        with open(data_path, "r", encoding="utf8") as f:
            data = json.load(f)
        
        # Data is a list of dictionaries now, each one a question and its associated answers and metadata.
        instances: List[Instance] = list()

        # UNCOMMENT BELOW FOR FEW-SHOT RUN
        # training_data_path: str = os.path.join("restricted", "mock_scs1.json")
        # assert os.path.exists(training_data_path)

        # with open(training_data_path, "r", encoding="utf8") as f:
        #     training_data = json.load(f)
        # for question in training_data:
        #     question_text = question['question']
        #     references = list()
        #     for index, answer in enumerate(question['options']):
        #         reference_answer = Output(text=answer)
        #         # Correct option offset by 1 due to zero-indexing
        #         tag = [CORRECT_TAG] if index == question['correct_option'] - 1 else []
        #         references.append(Reference(reference_answer, tags=tag))
        #     instance: Instance = Instance(
        #         input=Input(text=question_text),
        #         references=references,
        #         split=TRAIN_SPLIT,
        #     )
        #     instances.append(instance)
        
        for question in data:
            question_text = question['question']
            references = list()
            for index, answer in enumerate(question['options']):
                reference_answer = Output(text=answer)
                # Correct option offset by 1 due to zero-indexing
                tag = [CORRECT_TAG] if index == question['correct_option'] - 1 else []
                references.append(Reference(reference_answer, tags=tag))
            instance: Instance = Instance(
                input=Input(text=question_text),
                references=references,
                split=VALID_SPLIT, # Just doing zero shot to start
            )
            instances.append(instance)

        return instances
