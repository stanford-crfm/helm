import pandas as pd

from typing import List
from helm.common.general import check_file_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TEST_SPLIT,
)


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


def define_rules(target_specialty: str):
    rules = []
    rules.append(
        f"""- All data included from the notes, which is relevant for a specialty of {target_specialty}, is in the summary."""
    )
    rules.append(
        f"""- All assertions can be traced back to the notes; NEVER include assertions which cannot be traced back to the notes."""
    )
    rules.append(
        f"""- Information from the notes which is pertinent for a specialty of {target_specialty}, or potentially pertinent for a specialty of {target_specialty}, is NEVER omitted."""
    )
    rules.append(
        f"""- Information from the notes which is NOT pertinent for a specialty of {target_specialty} IS omitted from the summary."""
    )
    rules.append(f"""- The level of detail must be appropriate for a reader with a specialty of {target_specialty}.""")
    rules.append(
        f"""- All assertions must be made with logical order and grouping (temporal or systems/problem based)."""
    )
    rules.append(
        f"""- Summary must be comprehensible, using plain language that is completely familiar and well-structured for a reader with a specialty of {target_specialty}."""
    )
    rules.append(
        f"""- All assertions are captured with fewest words possible and without any redundancy in syntax or semantics."""
    )
    rules.append(
        f"""- Where applicable, go beyond relevant groups of events and generate reasoning over the events into a summary that is fully integrated for an overall clinical synopsis with prioritized information."""
    )
    rules.append(f"""- Avoid stigmatizing words as defined in guidelines and policy (OCR, NIDA, etc).""")
    rules.append(f"""- Keep the summary succinct; summarize all the notes in a single paragraph.""")
    rules.append(f"""- If there are medicine changes in the notes, mention them in the summary.""")
    rules.append(
        f"""- For every event (e.g., medicine change, new diagnosis, etc.) mentioned in your summary, mention WHEN it happened (communicate the timing of events) if that information is available in the note."""
    )
    rules.append(
        f"""- If it's unclear WHEN an event happened in the notes, instead explain that the event was mentioned by a note written at [timestamp of the note]."""
    )
    rules.append(
        f"""- For each SENTENCE in the summary, cite the <Note ID> source in the summary using the format <Note ID:IDVAL>, where IDVAL is the ID of the note."""
    )
    rules.append(
        f"""- Cite each note tag individually; when citing multiple notes, use the format <Note ID:IDVAL>, <Note ID:IDVAL>."""
    )
    rules.append(f"""- Prioritize citation order by relevance to the assertion.""")
    rules.append(f"""- Put the citations immediately after each sentence, where they are applicable.""")
    rules.append(f"""- NEVER group all the citations together on the last line.""")
    rules.append(f"""- ALL sentences MUST have a citation. ALL citations MUST be in <Note ID:IDVAL> format.""")
    rules.append(
        f"""- It is CRITICALLY IMPORTANT that you cite information to the note it came from! Wrongful citations are HARMFUL!"""
    )
    return rules


def get_notes(df: pd.DataFrame) -> str:
    prompt_notes = ""
    notes = []
    authors = []
    timestamps = []
    for _, row in df.iterrows():
        notes.append(row["text"])
        authors.append(row["note_type"])
        timestamps.append(row["charttime"])
    for i in range(len(notes)):
        prompt_notes += f"""<NoteID:{i+1}>
Written By: {authors[i]}
Timestamp: {timestamps[i]}
Note: {notes[i]}
<\\NoteID:{i+1}>
"""
    return prompt_notes


def create_prompt(df: pd.DataFrame) -> str:
    specialty = "emergency medicine"
    rules = define_rules(specialty)
    notes = get_notes(df)
    prompt = f"""You are an expert doctor.
Your task is to write a summary for a specialty of {specialty}, after reviewing a set of notes about a patient."""

    prompt += "\n\nRules for writing the summary:"

    for i in range(len(rules)):
        prompt += "\n" + rules[i]

    prompt += f"""\n\nSummarize the following <NoteSet>, which are presented to you in chronological order split by <Note ID>:

<NoteSet> 
{notes}
</NoteSet>
"""
    return prompt


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
    description = (
        "NoteSummary is a benchmark designed to evaluate clinical text generation. It pairs"
        "discharge summaries and radiology reports from MIMIC-IV with generation tasks"
        "such as writing discharge instructions or summarizing the brief hospital course. The"
        "benchmark assesses a model's ability to generate patient-facing documentation that is"
        "complete, empathetic, and clinically accurate."
    )
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
            prompt_di = create_prompt(df_admission)
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
