import pandas as pd
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


def create_prompt_text(clinical_note):
    # Create a prompt for the model to generate a care plan based on the clinical note
    prompt = f"""
You are provided with a clinical note regarding a physician-patient interaction. Your task is to \
extract specific information based solely on the content provided. Do not hallucinate or infer details \
that are not explicitly stated in the text. Any information you include must be directly entailed by the text.

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

Ensure your responses are concise, accurate, and entirely supported by the provided text. \
Do not introduce external knowledge or assumptions.

Clinical Note:
{clinical_note}
"""
    return prompt


class CHWCarePlanScenario(Scenario):
    """
    A scenario for a dataset containing free form text of a clinical health worker care plan, with the
    associated goal being to restructure that text into a given format.

    - Input:  The clinical note (column "MO Note").
    - Output:  The clinical note (column "MO Note"). We will use this note as the reference for entailment.
    """

    name = "chw_care_plan"
    description = (
        "NoteExtract is a benchmark that focuses on the structured extraction of information"
        "from free-form clinical text. It provides care plan notes authored by health workers"
        "and evaluates a model's ability to convert them into a predefined structured format,"
        "such as fields for Chief Complaint and History of Present Illness. The benchmark"
        "emphasizes faithful extraction without hallucination or inference."
    )
    tags = ["question_answering", "biomedical"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.data_path, msg=f"[CHWCarePlanScenario] Required data file not found: '{self.data_path}'")
        df = pd.read_csv(self.data_path)  # columns: ["text", "target", ...]

        instances: List[Instance] = []

        # Use the entire dataset as one split (TEST_SPLIT)
        for idx, row in df.iterrows():
            note_text: str = row["MO Note"]
            prompt_text = create_prompt_text(note_text)
            if pd.isna(note_text):
                print(f"Skipping row {idx} due to NaN value in 'MO Note'")
                continue
            # print(f"Prompt text: {prompt_text}")

            # Create one Instance per patient
            instance = Instance(
                input=Input(text=prompt_text),
                references=[Reference(Output(text=note_text), tags=[CORRECT_TAG])],
                split=TEST_SPLIT,
            )
            instances.append(instance)
        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name="chw_care_plan",
            display_name="NoteExtract",
            description="NoteExtract is a benchmark that focuses on the structured extraction of "
            "information from free-form clinical text. It provides care plan notes authored "
            "by health workers and evaluates a model's ability to convert them into a "
            "predefined structured format, such as fields for Chief Complaint and History "
            "of Present Illness. The benchmark emphasizes faithful extraction without "
            "hallucination or inference.",
            taxonomy=TaxonomyInfo(
                task="Text generation",
                what="Convert general text care plans into structured formats",
                when="Any",
                who="Clinician, Researcher",
                language="English",
            ),
            main_metric="chw_care_plan_accuracy",
            main_split="test",
        )
