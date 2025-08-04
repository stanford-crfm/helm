import pandas as pd

from typing import List
from helm.common.general import check_file_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
)
from helm.benchmark.scenarios.note_summary_scenario_helper import Summarizer


def file_preprocessing(data_path: str) -> pd.DataFrame:
    """
    Preprocess the data files to create a DataFrame with the necessary columns.
    task_objective: 'brief_hospital_course' or 'discharge_instructions'
    Use command to download: wget -r -N -c -np --user {PHYSIONET_USERNAME} \
    --ask-password https://physionet.org/files/discharge-me/1.3/
    data_path is directory that contains the downloaded files: '{base_dir}/physionet.org/'
    """
    # Load the first CSV file
    discharge_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/discharge.csv.gz"
    check_file_exists(
        discharge_path, msg=f"[NoteSummaryScenario] Required discharge file not found: '{discharge_path}'"
    )
    radiology_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/radiology.csv.gz"
    check_file_exists(
        radiology_path, msg=f"[NoteSummaryScenario] Required radiology file not found: '{radiology_path}'"
    )
    df_discharge = pd.read_csv(discharge_path, compression="gzip", keep_default_na=False)
    df_radiology = pd.read_csv(radiology_path, compression="gzip", keep_default_na=False)

    final_df = pd.concat([df_discharge, df_radiology], ignore_index=True)
    return final_df


class NoteSummaryScenario(Scenario):
    """
    NoteSummary is a discharge instruction generation dataset and brief hospital course generation \
    dataset collected from MIMIC-IV data.
    In this scenario, we only consider the discharge text as well as the radiology report text.
    We are using the phase I test set which is composed of 14,702 hospital admission instances.

    The splits are provided by the dataset itself.

    TASKS = {discharge instruction, brief hospital course}
    Sample Synthetic Prompt:
        Generate the {TASK} from the following patient discharge text and radiology report text.

        Discharge Text:
        Name: {Patient Name} Unit No: {Unit Number} Date of Birth: {DOB} Date of Admission:
        {DOA} Date of Discharge: {DOD}
        Chief Complaint: {Chief Complaint} History of Present Illness: {HPI} Past Medical History: {PMH}
        Medications on Admission: {Medications} Allergies: {Allergies} Physical Exam: {Physical Exam}
        Discharge Diagnosis: {Discharge Diagnosis}

        Radiology Report:
        {Radiology Report}

        {TASK}:
    @inproceedings{Xu_2024,
        title={ Discharge me: Bionlp acl’24 shared task on streamlining discharge documentation.},
        url={https://doi.org/10.13026/4a0k-4360},
        DOI={10.13026/27pt-1259},
        booktitle={ Proceedings of the 23rd Workshop on Biomedical Natural Language Processing (BioNLP) at ACL 2024},
        publisher={Association for Computational Linguistics},
        author={Xu, Justin and Delbrouck, Jean-Benoit and Johnston, Andrew and Blankemeier, Louis and Langlotz, Curtis},
        year={2024}
    }
    """

    name = "note_summary"
    description = "NoteSummary is a benchmark designed to evaluate clinical note summarization capabilities of LLMs."
    tags = ["biomedical"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        df = file_preprocessing(self.data_path)
        admissions = df["hadm_id"].unique()
        for admission in admissions:
            df_admission = df[df["hadm_id"] == admission]
            summarizer = Summarizer(
                notes=df_admission["text"].tolist(),
                authors=df_admission["note_type"].tolist(),
                timestamps=df_admission["charttime"].tolist(),
                target_specialty="emergency medicine",
            )
            prompt_di, _ = summarizer.build_prompt(anti_rules=0, omit_rules=0)
            instances.append(
                Instance(
                    input=Input(text=prompt_di),
                    references=[],
                    split=TEST_SPLIT,
                    extra_data={"notes": df_admission["text"].tolist()},
                )
            )

        return instances

    def read_file(self, file_path: str) -> List[str]:
        with open(file_path, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines
