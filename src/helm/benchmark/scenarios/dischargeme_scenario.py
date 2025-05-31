from typing import List
from helm.common.general import check_file_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
)
import pandas as pd


def file_preprocessing(data_path: str, task_objective: str) -> pd.DataFrame:
    """
    Preprocess the data files to create a DataFrame with the necessary columns.
    task_objective: 'brief_hospital_course' or 'discharge_instructions'
    Use command to download: wget -r -N -c -np --user {PHYSIONET_USERNAME} \
    --ask-password https://physionet.org/files/discharge-me/1.3/
    data_path is directory that contains the downloaded files: '{base_dir}/physionet.org/'
    """
    # Load the first CSV file
    diagnosis_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/diagnosis.csv.gz"
    check_file_exists(
        diagnosis_path, msg=f"[DischargeMeScenario] Required diagnosis file not found: '{diagnosis_path}'"
    )
    discharge_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/discharge.csv.gz"
    check_file_exists(
        discharge_path, msg=f"[DischargeMeScenario] Required discharge file not found: '{discharge_path}'"
    )
    target_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/discharge_target.csv.gz"
    check_file_exists(target_path, msg=f"[DischargeMeScenario] Required target file not found: '{target_path}'")
    radiology_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/radiology.csv.gz"
    check_file_exists(
        radiology_path, msg=f"[DischargeMeScenario] Required radiology file not found: '{radiology_path}'"
    )
    ed_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/edstays.csv.gz"
    check_file_exists(ed_path, msg=f"[DischargeMeScenario] Required ed file not found: '{ed_path}'")
    triage_path = f"{data_path}/files/discharge-me/1.3/test_phase_1/triage.csv.gz"
    check_file_exists(triage_path, msg=f"[DischargeMeScenario] Required triage file not found: '{triage_path}'")
    df_diagnosis = pd.read_csv(diagnosis_path, compression="gzip", keep_default_na=False)
    df_discharge = pd.read_csv(discharge_path, compression="gzip", keep_default_na=False)
    df_target = pd.read_csv(
        target_path,
        compression="gzip",
        keep_default_na=False,
    )
    df_radiology = pd.read_csv(radiology_path, compression="gzip", keep_default_na=False)
    df_ed = pd.read_csv(ed_path, compression="gzip", keep_default_na=False)
    df_triage = pd.read_csv(triage_path, compression="gzip", keep_default_na=False)
    df_diagnosis_triage = pd.merge(
        df_diagnosis, df_triage, on="subject_id", how="inner", suffixes=("_df_diagnosis", "_df_triage")
    )
    df_diagnosis_triage_discharge = pd.merge(
        df_diagnosis_triage, df_discharge, on="subject_id", how="inner", suffixes=("", "_df_discharge")
    )
    df_diagnosis_triage_discharge_radiology = pd.merge(
        df_diagnosis_triage_discharge, df_radiology, on="hadm_id", how="inner", suffixes=("", "_df_radiology")
    )

    df_features = pd.merge(
        df_diagnosis_triage_discharge_radiology, df_ed, on="hadm_id", how="inner", suffixes=("", "_df_ed")
    )

    # Reduce the DataFrame to remove duplicate hadm_id
    df_features_reduced = df_features.drop_duplicates(subset="hadm_id")
    columns_to_keep = ["text", "text_df_radiology", "hadm_id"]
    df_input = df_features_reduced[columns_to_keep]
    final_df = pd.merge(df_input, df_target, on="hadm_id", how="inner")

    def remove_substring(string, substring):
        return string.replace(substring, "")

    final_df["text"] = final_df.apply(lambda row: remove_substring(row["text"], row[task_objective]), axis=1)
    return final_df


def create_prompt(text: str, text_df_radiology: str, task_objective: str) -> str:
    """
    Create the prompt for the instance.
    """
    prompt = f"Generate the {task_objective} from the following patient discharge text and radiology report text.\
    \n\nDischarge Text:\n{text}\n\nRadiology Report:\n{text_df_radiology}\n\n{task_objective}:\n"
    return prompt


class DischargeMeScenario(Scenario):
    """
    DischargeMe is a discharge instruction generation dataset and brief hospital course generation \
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

    name = "dischargeme"
    description = "DischargeMe is a discharge instruction generation dataset and brief hospital course generation \
    dataset collected from MIMIC-IV data, consindering only the discharge text as well as the radiology report text."
    tags = ["biomedical"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        df_bhc = file_preprocessing(self.data_path, "brief_hospital_course")
        df_di = file_preprocessing(self.data_path, "discharge_instructions")

        for i in range(df_bhc.shape[0]):
            prompt_bhc = create_prompt(
                df_bhc.iloc[i]["text"], df_bhc.iloc[i]["text_df_radiology"], "Brief Hospital Course"
            )
            prompt_di = create_prompt(
                df_di.iloc[i]["text"], df_di.iloc[i]["text_df_radiology"], "Discharge Instructions"
            )
            answer_bhc = df_bhc.iloc[i]["brief_hospital_course"]
            answer_di = df_di.iloc[i]["discharge_instructions"]
            instances.append(
                Instance(
                    input=Input(text=prompt_bhc),
                    references=[Reference(Output(text=answer_bhc), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )
            instances.append(
                Instance(
                    input=Input(text=prompt_di),
                    references=[Reference(Output(text=answer_di), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances

    def read_file(self, file_path: str) -> List[str]:
        with open(file_path, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines
