import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)


# /share/pi/nigam/data/MedAlign on Carina
EHR_BASE_PATH = "/local-scratch/shahlab/aunell/helm/medalign_data/ehr_unzipped/full_ehrs"
INSTRUCTIONS_PATH = "/local-scratch/shahlab/aunell/helm/medalign_data/ehr-relevance-labels.csv"
CLINICIAN_RESPONSES_PATH = "/local-scratch/shahlab/aunell/helm/medalign_data/clinician-instruction-responses.csv"
PROMPT_TEMPLATES_BASE_PATH = "/local-scratch/shahlab/aunell/helm/medalign_data/prompt_templates/"


def extract_patient_id_from_fname(fname: str) -> Optional[int]:
    """
    Extracts and returns the patient ID from a given filename.

    The function expects filenames in the format 'EHR_<patient_id>.xml',
    where <patient_id> is a sequence of digits.

    Parameters:
        fname (str): The filename from which to extract the patient ID.

    Returns:
        Optional[int]: The extracted patient ID as an integer, or None if
                    the filename doesn't match the expected format.
    """
    regex_result = re.search(r"EHR_(\d+)\.xml", fname)
    if regex_result is None:
        return None
    return int(regex_result.group(1))


def get_ehrs(path_to_ehrs: str) -> Dict[int, str]:
    """
    Builds a map from Instruction ID to EHR (Electronic Health Record) timeline.

    EHR timelines are in string format and EHR files are read in from the
    user-specified directory. Each file in the directory should be named
    'EHR_<patient_id>.xml', where <patient_id> is a sequence of digits.

    See https://stanfordmedicine.box.com/s/r28wfwwude9rpjtu0szhzegmku8qv2pe

    Parameters:
        path_to_ehrs (str): The path to the directory containing the EHR files.

    Returns:
        Dict[int, str]: A dictionary mapping patient IDs to EHR timelines.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.isdir(path_to_ehrs):
        raise FileNotFoundError(f"The specified directory {path_to_ehrs} does not exist.")

    ehr_map = {}
    for fname in os.listdir(path_to_ehrs):
        pt_id = extract_patient_id_from_fname(fname)
        if pt_id is None:
            print(f"Warning: File '{fname}' does not match the expected format " "and will be skipped.")
            continue

        file_path = os.path.join(path_to_ehrs, fname)
        with open(file_path, encoding="utf-8", mode="r") as f:
            ehr = f.read()

        ehr_map[pt_id] = ehr
    return ehr_map


def get_instructions(path_to_instructions: str) -> Dict[int, Dict[str, Union[int, str]]]:
    """
    Builds map from Instruction ID to instruction details

    The needed information for creating the map is accomplished by reading
    a CSV file from the user-specified specified path.

    The CSV file is expected to contain at least the following columns:
    - instruction_id: The ID of the instruction.
    - question: The text of the instruction.
    - person_id: The ID of the associated patient.
    - is_selected_ehr: A flag indicating whether the instruction is selected.

    See https://stanfordmedicine.box.com/s/0om9qav2sklb9vaitn0ibye65vgbfx0e

    Parameters:
        path_to_instructions (str): Path to CSV file containing instructions.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary mapping instruction IDs to a
            dictionary containing instruction text and associated patient ID.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file does not contain the expected columns.
    """
    if not os.path.exists(path_to_instructions):
        raise FileNotFoundError(f"The specified file {path_to_instructions} does not exist.")

    instructions_df = pd.read_csv(path_to_instructions)
    required_columns = {
        "instruction_id",
        "question",
        "person_id",
        "is_selected_ehr",
    }
    if not required_columns.issubset(instructions_df.columns):
        raise ValueError(f"The CSV file is missing one or more of the required columns: {required_columns}")

    selected_instructions_df = instructions_df.query("is_selected_ehr == 'yes'")
    instructions_map = {
        row["instruction_id"]: {
            "instruction": row["question"],
            "patient_id": row["person_id"],
        }
        for _, row in selected_instructions_df.iterrows()
    }
    return instructions_map


class MedAlignScenario(Scenario):
    """Scenario defining the MedAlign task as defined in the following work by Fleming et al:
    @article{fleming2023medalign,
    title={MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records},
    author={Scott L. Fleming
        and Alejandro Lozano
        and William J. Haberkorn
        and Jenelle A. Jindal
        and Eduardo P. Reis
        and Rahul Thapa
        and Louis Blankemeier
        and Julian Z. Genkins
        and Ethan Steinberg
        and Ashwin Nayak
        and Birju S. Patel
        and Chia-Chun Chiang
        and Alison Callahan
        and Zepeng Huo
        and Sergios Gatidis
        and Scott J. Adams
        and Oluseyi Fayanju
        and Shreya J. Shah
        and Thomas Savage
        and Ethan Goh
        and Akshay S. Chaudhari
        and Nima Aghaeepour
        and Christopher Sharp
        and Michael A. Pfeffer
        and Percy Liang
        and Jonathan H. Chen
        and Keith E. Morse
        and Emma P. Brunskill
        and Jason A. Fries
        and Nigam H. Shah},
    journal={arXiv preprint arXiv:2308.14089},
    year={2023}
    }

    Each instance includes:
    - input: the instruction and patient record
    - reference: the clinical 'gold standard' completion for the instruction for the given patient record

    This is a clinical instruction-following task, wherein a generative language model must follow
    the instructions using the provided patient record. As explained in the MedAlign work, each example
    is guaranteed to be completable for the given patient record.

    This task is evaluated using COMET and BERTScore metrics.
    """

    name = "medalign"
    description = "MedAlign clinical instruction following task and dataset"
    tags = ["instruction_following", "generation"]

    def __init__(self, prompt_template: str = "generic.txt"):
        super().__init__()
        self.prompt_template = prompt_template

    def _load_dataset(self) -> Tuple[Dict[int, Dict[str, Union[int, str]]], Dict[int, str], pd.DataFrame]:
        assert os.path.exists(INSTRUCTIONS_PATH)
        assert os.path.exists(CLINICIAN_RESPONSES_PATH)
        assert os.path.exists(EHR_BASE_PATH)

        instructions = get_instructions(INSTRUCTIONS_PATH)
        ehrs = get_ehrs(EHR_BASE_PATH)
        gold_df = pd.read_csv(CLINICIAN_RESPONSES_PATH)

        # required filtering to match MedAlign code.
        # TODO: clean this up either in the data files or with better logic
        gold_df = gold_df[gold_df.annotator_num == "Annotator_1"]
        return instructions, ehrs, gold_df

    def get_instances(self, output_path: str) -> List[Instance]:
        instructions, ehrs, clinician_responses_df = self._load_dataset()
        prompt_template_path = Path(PROMPT_TEMPLATES_BASE_PATH) / self.prompt_template
        if not (prompt_template_path.exists() and prompt_template_path.is_file()):
            raise RuntimeError(f"Prompt template path {str(prompt_template_path)} not found!")

        with open(prompt_template_path, "r", encoding="utf-8") as fh:
            prompt_template = fh.read()

        instances: List[Instance] = []
        for instruction_id, instruction_dict in instructions.items():
            # get the actual instruction
            instruction: Union[str, int] = instruction_dict["instruction"]  # question or task

            # get the patient EHR selected for this instruction
            pt_id: Union[str, int] = instruction_dict["patient_id"]
            relevant_ehr = ehrs[pt_id]  # type: ignore

            # get the clinican response which serves as the reference
            clinician_response_rows = list(
                clinician_responses_df[clinician_responses_df.instruction_id == instruction_id].iterrows()
            )
            assert len(clinician_response_rows) == 1
            clinician_response = clinician_response_rows[0][1].clinician_response

            instances.append(
                Instance(
                    input=Input(
                        text=instruction,  # type: ignore
                    ),
                    references=[Reference(Output(clinician_response), tags=[CORRECT_TAG])],
                    extra_data={
                        "prompt_template": prompt_template,
                        "ehr": relevant_ehr,
                    },
                    split=TEST_SPLIT,
                )
            )
        return instances
