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
        "MedMCQA is a multiple-choice question answering (MCQA) dataset designed to address "
        "real-world medical entrance exam questions."
    )
    tags = ["question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path: str = os.path.join(output_path, "data.json")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            # unpack=True,
            # unpack_type="unzip",
        )

        instances: List[Instance] = []

        print("DATA PATH: ", data_path)
        with open(data_path, "r") as f:
            example = json.loads(f.read(), strict=False) # to allow newline control characters
        print("EXAMPLE: ", example)
        references = list()
        for key in example:
            if key[:2] == 'op':
                # CURRENTLY HARDCODED CORRECT CHOICE FOR TESTING NEED TO UPDATE
                references.append(Reference(Output(text=example[key]), tags=[CORRECT_TAG] if key == 'opb' else []))
        instance: Instance = Instance(
                    input=Input(text=example["question"]),
                    references=references,
                    split=VALID_SPLIT, # No training data for testing currently
                )

        # # From https://github.com/MedMCQA/MedMCQA#model-submission-and-test-set-evaluation,
        # # "to preserve the integrity of test results, we do not release the test set's ground-truth to the public".
        # for split in [TRAIN_SPLIT, VALID_SPLIT]:
        #     # Although the files end with ".json", they are actually JSONL files
        #     split_file_name: str = f"{'dev' if split == VALID_SPLIT else split}.json"
        #     split_path: str = os.path.join(data_path, split_file_name)

        #     with open(split_path, "r") as f:
        #         for line in f:
        #             # The data fields and their explanations can be found here:
        #             # https://github.com/MedMCQA/MedMCQA#data-fields
        #             example: Dict[str, str] = json.loads(line.rstrip())

        #             # Just edit my references to use the format given
        #             # HELM does not look at the JSON, it can be formatted however
        #             references: List[Reference] = [
        #                 # Value of "cop" corresponds to the index of the correct option
        #                 Reference(Output(text=example[option]), tags=[CORRECT_TAG] if index == example["cop"] else [])
        #                 for option, index in MedMCQAScenario.ANSWER_OPTION_TO_INDEX.items()
        #             ]
        #             instance: Instance = Instance(
        #                 input=Input(text=example["question"]),
        #                 references=references,
        #                 split=split,
        #             )
        #             instances.append(instance)

        return instances
