import os
import pandas as pd
import numpy as np
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

def create_prompt_text(clinical_note):
    # Create a prompt for the model to generate a care plan based on the clinical note
    prompt = f"""
You are provided with a clinical note regarding a physician-patient interaction. Your task is to extract specific information based solely on the content provided. Do not hallucinate or infer details that are not explicitly stated in the text. Any information you include must be directly entailed by the text.

Instructions:

Extract the required information precisely as presented in the source text.

If the text does not contain specific information, clearly state "Not mentioned".

Maintain the patient's original wording whenever possible.

Response Format:

Chief Complaint
[ENTER CHIEF COMPLAINT]

History of Present Illness

Onset: When did it start? Did it begin suddenly or gradually? [ENTER ONSET INFORMATION]

Provoking/Palliating Factors: What makes the symptoms better or worse? [ENTER Provoking/Palliating Factors INFORMATION]

Quality: Describe the symptoms, e.g., sharp pain, dull pain, stabbing pain. [ENTER QUALITY INFORMATION]

Region/Radiation: Where are your symptoms located? Do they move? [ENTER REGION/RADIATION INFORMATION]

Severity: On a scale of 1 to 10, how severe are your symptoms? [ENTER SEVERITY INFORMATION]

Timing: When do you experience the symptoms? What times of day? [ENTER TIMING INFORMATION]

Related Symptoms: Are there any other symptoms related to the main complaint? [ENTER RELATED SYMPTOMS INFORMATION]

Ensure your responses are concise, accurate, and entirely supported by the provided text. Do not introduce external knowledge or assumptions.

Clinical Note:
{clinical_note}
"""
    return prompt


class CHWCarePlanScenario(Scenario):
    """
    A scenario for MIMIC-IV discharge summaries where the task is to predict the ICD-10 code(s).

    - Input:  The clinical note (column "MO Note").
    - Output:  The clinical note (column "MO Note"). We will use this note as the reference for entailment.
    """

    name = "chw_care_plan"
    description = "Given a clinical note, extract key information into a given format."
    tags = ["question_answering", "biomedical"]

    def __init__(self):
        """
        :param data_file: Path to the mimiciv_icd10.feather file.
        """
        super().__init__()
        self.data_file = "/share/pi/nigam/datasets/CHW_Dataset.csv"
        
    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(os.path.dirname(self.data_file))
        
        df = pd.read_csv(self.data_file)  # columns: ["text", "target", ...]

        instances: List[Instance] = []
        
        # Use the entire dataset as one split (TEST_SPLIT)
        for idx, row in df.iterrows():
            try:
                note_text: str = row["MO Note"]
                prompt_text = create_prompt_text(note_text)
                # print(f"Prompt text: {prompt_text}")

                # Create one Instance per patient
                instance = Instance(
                    input=Input(text=prompt_text),
                    references=[
                        Reference(
                            Output(text=note_text),
                            tags=[CORRECT_TAG]
                        )
                    ],
                    split=TEST_SPLIT
                )
                instances.append(instance)
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        return instances