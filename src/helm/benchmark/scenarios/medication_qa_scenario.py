import os
from typing import List

import pandas as pd

from helm.common.general import ensure_file_downloaded

from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Input, Instance, Output, Reference, Scenario


class MedicationQAScenario(Scenario):
    """
    The gold standard corpus for medication question answering introduced in the MedInfo 2019 paper
    "Bridging the Gap between Consumers’ Medication Questions and Trusted Answers":
    http://ebooks.iospress.nl/publication/51941

    This dataset has consumer questions, as opposed to very clinical questions.

    Paper citation:

        @inproceedings{BenAbacha:MEDINFO19,
        author    = {Asma {Ben Abacha} and Yassine Mrabet and Mark Sharp and
                    Travis Goodwin and Sonya E. Shooshan and Dina Demner{-}Fushman},
        title     = {Bridging the Gap between Consumers’ Medication Questions and Trusted Answers},
        booktitle = {MEDINFO 2019},
        year      = {2019},
        }
    """

    SOURCE_REPO_URL = "https://github.com/abachaa/Medication_QA_MedInfo2019/raw/master/"
    FILENAME = "MedInfo2019-QA-Medications.xlsx"

    name = "medication_qa"
    description = "Open text question-answer pairs regarding consumer health questions about medication."
    tags = ["knowledge", "generation", "question_answering", "biomedical"]

    def download_medication_qa(self, path: str):
        """download the .xlsx spreadsheet containing the question-answer pairs"""
        ensure_file_downloaded(
            source_url=os.path.join(self.SOURCE_REPO_URL, self.FILENAME),
            target_path=os.path.join(path, self.FILENAME),
            unpack=False,
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        self.download_medication_qa(output_path)
        data_path = os.path.join(output_path, self.FILENAME)

        data = pd.read_excel(data_path)
        data = data[~data.Answer.isna()]  # remove rows missing answers
        instances = [
            Instance(
                input=Input(row.Question),
                references=[Reference(Output(row.Answer), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            for _, row in data.iterrows()
        ]

        return instances
