import os
import pandas as pd
from typing import List

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
)


class CLEARScenario(Scenario):
    """
    CLEARScenario is a dataset of human-labeled medical texts that indicate whether a patient has a history
    of alcohol dependence. Each example includes:

      - text: A medical note or patient report.
      - result_human: The human-provided label where:
            1 indicates the patient has a history of alcohol dependence,
            0 indicates the patient does not have a history of alcohol dependence,
            2 indicates uncertainty about the patient's history of alcohol dependence.
            
    For this scenario, the human label is mapped to a multiple-choice option as follows:
            1 -> A, 0 -> B, 2 -> C

    The task is to classify the text using a multiple-choice format.

    Sample Synthetic Prompt:
        You are a helpful medical assistant. Determine whether the patient has a history of alcohol dependence.

        Text: [insert text here]

        A. Has a history of alcohol dependence
        B. Does not have a history of alcohol dependence
        C. Uncertain

        Answer:
    """

    name = "clear_alcohol_dependence"
    description = "A dataset for classifying whether a patient has a history of alcohol dependence based on human-labeled medical texts from CLEAR."
    tags = ["classification", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        excel_path = "/share/pi/nigam/suhana/medhelm/data/CLEAR/human_labeled/alcohol_dependence.xlsx"
        ensure_directory_exists(os.path.dirname(excel_path))

        df = pd.read_excel(excel_path)

        instances: List[Instance] = []
        # All instances are assigned to the test split (zero-shot setting)
        split = TEST_SPLIT

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            label = str(row.get("result_human", "")).strip()

            # Skip the instance if any required field is missing
            if not text or not label:
                continue

            label = {"1": "A", "0": "B", "2": "C"}.get(label, label)
            # Construct the input prompt
            input_text = (
                "You are a helpful medical assistant. Determine whether the patient has a history of alcohol dependence. Answer A for yes, B for no, C for maybe. Do not add any other text, punctuation, or symbols \n\n"
                "Original Clinical Note:\n"
                f"{text}\n\n"
                "Answer A for yes, B for no, C for maybe. Do not add any other text, punctuation, or symbols"
            )

            instances.append(
                Instance(
                    input=Input(text=input_text),
                    references=[Reference(Output(text=label), tags=[CORRECT_TAG])],
                    split=split,
                )
            )

        return instances
