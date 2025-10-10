import pandas as pd
import numpy as np
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import check_file_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
    ScenarioMetadata,
)


class MIMICIVBillingCodeScenario(Scenario):
    """
    A scenario for MIMIC-IV discharge summaries where the task is to predict the ICD-10 code(s).

    - Input:  The clinical note (column "text").
    - Output: The list of ICD-10 codes (column "target").
    """

    name = "mimiciv_billing_code"
    description = (
        "MIMIC-IV Billing Code is a benchmark derived from discharge summaries in the"
        "MIMIC-IV database, paired with their corresponding ICD-10 billing codes. The task"
        "requires models to extract structured billing codes based on free-text clinical notes,"
        "reflecting real-world hospital coding tasks for financial reimbursement."
    )
    tags = ["question_answering", "biomedical"]

    def __init__(self, data_path: str):
        """
        :param data_path: Path to the mimiciv_icd10.feather file.
        """
        super().__init__()
        self.data_path = data_path

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(
            self.data_path, msg=f"[MIMICIVBilligCodeScenario] Required data file not found: '{self.data_path}'"
        )

        # Read the preprocessed MIMIC-IV data (.feather format)
        df = pd.read_feather(self.data_path)  # columns: ["text", "target", ...]

        instances: List[Instance] = []

        # Use the entire dataset as one split (TEST_SPLIT)
        for idx, row in df.iterrows():
            try:
                note_text: str = row["text"]
                icd10_codes = row["target"]

                # Convert numpy array to list if necessary
                if isinstance(icd10_codes, np.ndarray):
                    icd10_codes = icd10_codes.tolist()
                elif not isinstance(icd10_codes, list):
                    icd10_codes = [str(icd10_codes)]  # Handle single values

                # Convert all codes to strings and join
                codes_str = ",".join(str(code) for code in icd10_codes)

                # Create one Instance per row
                instance = Instance(
                    input=Input(text=note_text),
                    references=[Reference(Output(text=codes_str), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
                instances.append(instance)
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue

        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name="mimiciv_billing_code",
            display_name="MIMIC-IV Billing Code",
            description="MIMIC-IV Billing Code is a benchmark derived from discharge summaries in the "
            "MIMIC-IV database, paired with their corresponding ICD-10 billing codes. The "
            "task requires models to extract structured billing codes based on free-text "
            "clinical notes, reflecting real-world hospital coding tasks for financial "
            "reimbursement.",
            taxonomy=TaxonomyInfo(
                task="Classification",
                what="Predict ICD-10 billing codes from clinical discharge notes",
                when="During or after patient discharge",
                who="Hospital Admistrator",
                language="English",
            ),
            main_metric="mimiciv_billing_code_f1",
            main_split="test",
        )
