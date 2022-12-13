import csv
import os
import re
import string
from typing import Dict, List
import pyreadstat
from nltk import pos_tag
import numpy as np
import pandas as pd

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG


class SurveyScenario(Scenario):
    name = "surveys"
    description = "Surveys"
    tags = ["multiple_choice"]

    def __init__(self, survey_type: str, train_type: str):
        self.survey_type: str = survey_type
        self.train_type: str = train_type
            
    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path: str = os.path.join(self.output_path, "data")
        # ensure_file_downloaded(
        #    source_url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        #    target_path=data_path,
        #    unpack=True,
        #    unpack_type="untar",
        # )

        # Read all the instances
        instances: List[Instance] = []
        splits: Dict[str, str] = {
            "auxiliary_train": TRAIN_SPLIT,
            "dev": TRAIN_SPLIT,
            "val": VALID_SPLIT,
            "test": TEST_SPLIT,
        }

        
        for split in ["dev", "test"]:
            
            
            if split == "dev":
                if self.train_type == "None": continue
                csv_path: str = os.path.join(data_path, f"{self.train_type}.csv")
            else:
                csv_path: str = os.path.join(data_path, f"{self.survey_type}.csv")
                    
            ensure_directory_exists(csv_path)
            
            df = pd.read_csv(csv_path, sep="\t")
            df["options"] = df.apply(lambda x: eval(x["options"]), axis=1)

            for qidx, (question, answers) in enumerate(zip(df["question"], df["options"])):

                prefixes = list(string.ascii_uppercase)
                assert len(prefixes) >= len(answers)
                answers_dict = dict(zip(prefixes, answers))
                # Ignore, there is no correct answer
                correct_answer = answers[0] if split == "test" \
                                else df['correct'][qidx]  

                def answer_to_reference(answer):
                    return Reference(output=answer, tags=[CORRECT_TAG] if answer == correct_answer else [])

                instance = Instance(
                    input=question,
                    references=list(map(answer_to_reference, answers)),
                    split=splits[split],
                )
                instances.append(instance)

        return instances
