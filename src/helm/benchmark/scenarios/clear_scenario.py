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
    description = (
        "A dataset for evaluating alcohol dependence history "
        "detection from patient notes with yes/no/maybe classifications."
    )
    tags = ["classification", "biomedical"]

    POSSIBLE_ANSWER_CHOICES: List[str] = [
        "Has a history of alcohol dependence",
        "Does not have a history of alcohol dependence",
        "Uncertain",
    ]

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

            # Map the human label to the multiple-choice option: 1 -> A, 0 -> B, 2 -> C
            mapped_label = {
                "1": "Has a history of alcohol dependence",
                "0": "Does not have a history of alcohol dependence",
                "2": "Uncertain",
            }.get(label, label)

            # Build the References using the possible answer choices,
            # marking the correct one with the CORRECT_TAG.
            references: List[Reference] = [
                Reference(Output(text=choice), tags=[CORRECT_TAG] if choice == mapped_label else [])
                for choice in self.POSSIBLE_ANSWER_CHOICES
            ]

            input_text = (
                "You are a helpful medical assistant."
                "Determine whether the patient has a history of alcohol dependence. Answer 'A', 'B', or 'C'. \n\n"
                "Original Clinical Note:\n"
                f"{text}\n\n"
            )

            instances.append(
                Instance(
                    input=Input(text=input_text),
                    references=references,
                    split=split,
                )
            )

        return instances
