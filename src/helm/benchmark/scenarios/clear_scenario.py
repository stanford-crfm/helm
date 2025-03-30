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
    of various medical conditions like alcohol dependence, depression, PTSD etc. Each example includes:

      - text: A medical note or patient report.
      - result_human: The human-provided label where:
            1 indicates the patient has a history of the condition,
            0 indicates the patient does not have a history of the condition,
            2 indicates uncertainty about the patient's history of the condition.

    For this scenario, the human label is mapped to a multiple-choice option as follows:
            1 -> A, 0 -> B, 2 -> C

    The task is to classify the text using a multiple-choice format.

    Sample Synthetic Prompt:
        You are a helpful medical assistant. Determine whether the patient has a history of <medical condition>.

        Text: [insert text here]

        A. Has a history of alcohol dependence
        B. Does not have a history of alcohol dependence
        C. Uncertain

        Answer:
    """

    # List of all available conditions
    CONDITIONS = [
        "alcohol_dependence",
        "attention_deficit_hyperactivity_disorder",
        "bipolar_disorder",
        "chronic_pain",
        "homelessness",
        "liver_disease",
        "major_depression",
        "personality_disorder",
        "post_traumatic_stress_disorder",
        "substance_use_disorder",
        "suicidal_behavior",
        "tobacco_dependence",
        "unemployment",
    ]

    # Map condition names to human-readable descriptions for prompts
    CONDITION_PROMPTS = {
        "alcohol_dependence": "alcohol dependence",
        "attention_deficit_hyperactivity_disorder": "attention deficit hyperactivity disorder (ADHD)",
        "bipolar_disorder": "bipolar disorder",
        "chronic_pain": "chronic pain",
        "homelessness": "homelessness",
        "liver_disease": "liver disease",
        "major_depression": "major depression",
        "personality_disorder": "personality disorder",
        "post_traumatic_stress_disorder": "post-traumatic stress disorder (PTSD)",
        "substance_use_disorder": "substance use disorder",
        "suicidal_behavior": "suicidal behavior",
        "tobacco_dependence": "tobacco dependence",
        "unemployment": "unemployment",
    }

    def __init__(self, condition: str):
        """Initialize the scenario with a specific medical condition"""
        super().__init__()

        if condition not in self.CONDITIONS:
            raise ValueError(f"Condition '{condition}' not supported. Available conditions: {self.CONDITIONS}")

        self.condition = condition
        self.name = f"clear_{condition}"
        self.description = f"A dataset for evaluating {self.CONDITION_PROMPTS[condition]} detection from patient notes with yes/no/maybe classifications."  # noqa: E501
        self.tags = ["classification", "biomedical", condition.replace("_", "-")]

    def get_answer_choices(self) -> List[str]:
        """Get the possible answer choices with the condition filled in."""
        condition_text = self.CONDITION_PROMPTS[self.condition]
        return [f"Has a history of {condition_text}", f"Does not have a history of {condition_text}", "Uncertain"]

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load and process the data for the specified conditon."""
        data_dir = "/share/pi/nigam/suhana/medhelm/data/CLEAR/human_labeled/"
        excel_path = os.path.join(data_dir, f"{self.condition}.xlsx")
        ensure_directory_exists(os.path.dirname(excel_path))

        df = pd.read_excel(excel_path)

        possible_answer_choices = self.get_answer_choices()

        instances: List[Instance] = []
        # All instances are assigned to the test split (zero-shot setting)
        split = TEST_SPLIT
        condition_text = self.CONDITION_PROMPTS[self.condition]

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            label = str(row.get("result_human", "")).strip()

            # Skip the instance if any required field is missing
            if not text or not label:
                continue

            # Map the human label to the multiple-choice option: 1 -> A, 0 -> B, 2 -> C
            label_mapping = {
                "1": f"Has a history of {condition_text}",
                "0": f"Does not have a history of {condition_text}",
                "2": "Uncertain",
            }
            mapped_label = label_mapping.get(label, label)

            # Build the References using the possible answer choices,
            # marking the correct one with the CORRECT_TAG.
            references: List[Reference] = [
                Reference(Output(text=choice), tags=[CORRECT_TAG] if choice == mapped_label else [])
                for choice in possible_answer_choices
            ]

            input_text = (
                f"You are a medical assistant reviewing patient notes. "
                f"Determine whether the patient has a history of {condition_text}.\n\n"
                f"Original Clinical Note:\n"
                f"{text}\n\n"
                f"A. Has a history of {condition_text}\n"
                f"B. Does not have a history of {condition_text}\n"
                f"C. Uncertain\n\n"
                f"Answer:"
            )

            instances.append(
                Instance(
                    input=Input(text=input_text),
                    references=references,
                    split=split,
                )
            )

        return instances
