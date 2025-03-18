import os
import re

from typing import Any, Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    Input,
    Scenario,
    Instance,
    CORRECT_TAG,
    Reference,
    Output,
)

ORIGINAL_DEFINITIONS = {
    "ABDOMINAL": "History of intra-abdominal surgery, small or large intestine resection, or small bowel obstruction",
    "ADVANCED-CAD": "Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” \
    as having 2 or more of the following: • Taking 2 or more medications to treat CAD • \
    History of myocardial infarction (MI) • Currently experiencing angina • Ischemia, past or present",
    "ALCOHOL-ABUSE": "Current alcohol use over weekly recommended limits",
    "ASP-FOR-MI": "Use of aspirin for preventing myocardial infarction (MI)",
    "CREATININE": "Serum creatinine level above the upper normal limit",
    "DIETSUPP-2MOS": "Taken a dietary supplement (excluding vitamin D) in the past 2 months",
    "DRUG-ABUSE": "Current or past history of drug abuse",
    "ENGLISH": "Patient must speak English",
    "HBA1C": "Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%",
    "KETO-1YR": "Diagnosis of ketoacidosis within the past year",
    "MAJOR-DIABETES": "Major diabetes-related complication. For the purposes of this annotation, we define \
    “major complication” (as opposed to “minor complication”) as any of the following that are a result of \
    (or strongly correlated with) uncontrolled diabetes: • Amputation • Kidney damage • Skin conditions • \
    Retinopathy • nephropathy • neuropathy",
    "MAKES-DECISIONS": "Patient must make their own medical decisions",
    "MI-6MOS": "Myocardial infarction (MI) within the past 6 months",
}
# Custom definitions for better prompts
LONG_DEFINITIONS = {
    "ABDOMINAL": "History of intra-abdominal surgery. This could include any form of intra-abdominal surgery, \
    including but not limited to small/large intestine resection or small bowel obstruction",
    "ADVANCED-CAD": "Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define \
    “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History \
    of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. \
    The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient \
    does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just \
    have cardiovascular disease, then they do not meet this criterion.",
    "ALCOHOL-ABUSE": "Current alcohol use over weekly recommended limits",
    "ASP-FOR-MI": "Use of aspirin for preventing myocardial infarction (MI)..",
    "CREATININE": "Serum creatinine level above the upper normal limit",
    "DIETSUPP-2MOS": "Consumption of a dietary supplement (excluding vitamin D) in the past 2 months. To assess \
    this criterion, go through the list of medications_and_supplements taken from the note. If a substance could \
    potentially be used as a dietary supplement (i.e. it is commonly used as a dietary supplement, even if it \
    is not explicitly stated as being used as a dietary supplement), then the patient meets this criterion. \
    Be lenient and broad in what is considered a dietary supplement. For example, a 'multivitamin' and \
    'calcium carbonate' should always be considered a dietary supplement if they are included in this list.",
    "DRUG-ABUSE": "Current or past history of drug abuse",
    "ENGLISH": "Patient speaks English. Assume that the patient speaks English, unless otherwise explicitly noted. \
    If the patient's language is not mentioned in the note, then assume they speak English and thus meet \
    this criteria.",
    "HBA1C": "Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%",
    "KETO-1YR": "Diagnosis of ketoacidosis within the past year",
    "MAJOR-DIABETES": "Major diabetes-related complication. Examples of “major complication” (as opposed to \
    “minor complication”) include, but are not limited to, any of the following that are a result of (or \
    strongly correlated with) uncontrolled diabetes: • Amputation • Kidney damage • Skin conditions • Retinopathy \
    • nephropathy • neuropathy. Additionally, if multiple conditions together imply a severe case of diabetes, \
    then count that as a major complication.",
    "MAKES-DECISIONS": "Patient must make their own medical decisions. Assume that the patient makes their own \
    medical decisions, unless otherwise explicitly noted. There is no information provided about the \
    patient's ability to make their own medical decisions, then assume they do make their own decisions and \
    therefore meet this criteria.\"",
    "MI-6MOS": "Myocardial infarction (MI) within the past 6 months",
}


class XMLDataLoader:
    def __init__(
        self, path_to_folder: str, is_convert_to_numbers=True, is_split_text=True, is_remove_excessive_new_lines=True
    ):
        self.path_to_folder = path_to_folder
        self.is_convert_to_numbers = is_convert_to_numbers
        self.is_split_text = is_split_text
        self.is_remove_excessive_new_lines = is_remove_excessive_new_lines

    def load_data(self) -> List[Dict[str, Any]]:
        """Main function: Data loader for the XML files"""
        data = []
        file_names = os.listdir(self.path_to_folder)
        file_names = sorted([file for file in file_names if file.endswith(".xml")])
        for file_name in file_names:
            file_path = os.path.join(self.path_to_folder, file_name)
            text, labels = self.parse_xml(file_path)
            data.append({"patient_id": file_name.replace(".xml", ""), "ehr": text, "labels": labels})

        return data

    @staticmethod
    def get_date_of_note(patient: Dict[str, Any], note_idx: int) -> Optional[str]:
        """Get date of note for patient"""
        assert note_idx <= len(patient["ehr"]), f"{note_idx} out of bounds for {patient['patient_id']}"
        note: str = patient["ehr"][note_idx]
        match = re.search(r"Record date: (\d{4}-\d{2}-\d{2})", note)
        date = match.group(1) if match else None
        if not date:
            print(f"ERROR - Could not find the date for patient {patient['patient_id']}")
        return date

    @staticmethod
    def get_current_date_for_patient(patient: Dict[str, Any]) -> Optional[str]:
        """Get most recent date visible in files for a given patient"""
        most_recent_date = None
        for note in patient["ehr"]:
            match = re.search(r"Record date: (\d{4}-\d{2}-\d{2})", note)
            most_recent_date = match.group(1) if match else most_recent_date
        if not most_recent_date:
            print(f"ERROR - Could not find the date for patient {patient['patient_id']}")
        return most_recent_date

    def parse_xml(self, XML_file) -> Tuple[List[str], Dict[str, str]]:
        tree = ET.parse(XML_file)
        root = tree.getroot()
        text_content = ""
        result_text: List[str] = []
        tags = {}
        for elem in root.iter():
            if elem.tag == "TEXT":
                text_content = elem.text if elem.text else ""
                if self.is_remove_excessive_new_lines:
                    text_content = self.remove_excessive_newlines(text_content)
                if self.is_split_text:
                    result_text = self.split_text(text_content)
                else:
                    result_text = [text_content]
            elif elem.tag == "TAGS":
                tags = self.read_tags(root)
        return (result_text, tags)

    def read_tags(self, root) -> Dict[str, str]:
        """Reads the tags from an XML file and returns a dictionary of tags"""
        tags_dict = {}
        for tag in root.iter("TAGS"):
            for subtag in tag:
                met_value = subtag.attrib.get("met")
                if self.is_convert_to_numbers:
                    met_value = 1 if met_value == "met" else 0
                tags_dict[subtag.tag] = met_value
        return tags_dict

    def split_text(self, text: str) -> List[str]:
        split_char = "*" * 100
        parts = [x.strip() for x in text.split(split_char) if x.strip() != ""]
        return parts

    def remove_excessive_newlines(self, text: str) -> str:
        text = text.replace("\n\n\n", "\n")
        return text


class N2C2CTMatchingScenario(Scenario):
    """
    From "Cohort selection for clinical trials: n2c2 2018 shared task track 1" (Stubbs et al. 2019).
    N2C2 is a collection of 288 patients (202 train / 86 test), each with 2-5 deidentified real-world clinical notes.
    We use the prompt LLM formulation from Wornow et al. (2024).

    Citation
    ```
    @article{stubbs2019cohort,
        title={Cohort selection for clinical trials: n2c2 2018 shared task track 1},
        author={Stubbs, Amber and Filannino, Michele and Soysal, Ergin and Henry, Samuel and Uzuner, {\"O}zlem},
        journal={Journal of the American Medical Informatics Association},
        volume={26},
        number={11},
        pages={1163--1171},
        year={2019},
        publisher={Oxford University Press}
    }
    @article{wornow2024zero,
        title={Zero-shot clinical trial patient matching with llms},
        author={Wornow, Michael and Lozano, Alejandro and Dash, Dev and Jindal, Jenelle and Mahaffey, \
        Kenneth W and Shah, Nigam H},
        journal={NEJM AI},
        pages={AIcs2400360},
        year={2024},
        publisher={Massachusetts Medical Society}
    }
    ```
    """

    name = "n2c2_ct_matching"
    description = "A dataset that provides clinical notes and asks the model to classify whether the \
    patient is a valid candidate for a provided clinical trial."
    tags = []  # TODO

    POSSIBLE_ANSWER_CHOICES: List[str] = [
        "yes",
        "no",
    ]

    def __init__(self, subject: str):
        super().__init__()
        self.subject: str = subject  # specific inclusion criterion to assess
        self.path_to_train_dir: str = "/share/pi/nigam/data/medhelm/n2c2_ct_matching/train/"
        self.path_to_test_dir: str = "/share/pi/nigam/data/medhelm/n2c2_ct_matching/test/"

    def create_prompt(self, patient: Dict[str, Any]) -> str:
        # Cast None values to empty strings during string formatting, but keep the original functions returning None
        notes_list = [
            f"## Note #{i+1}\nDate: {XMLDataLoader.get_date_of_note(patient, i) or ''}\n{note}"
            for i, note in enumerate(patient["ehr"])
        ]
        notes: str = ("\n" + "*" * 50 + "\n\n").join(notes_list)
        current_date = XMLDataLoader.get_current_date_for_patient(patient)
        prompt = f"""
    # Task
    Your job is to decide whether the given patient meets the inclusion criterion for a clinical trial.

    # Inclusion Criterion
    The inclusion criterion being assessed is: "{self.subject}".
    The definition of the inclusion criterion is: "{LONG_DEFINITIONS[self.subject]}".

    # Patient Clinical Notes
    Below is a set of {len(patient['ehr'])} clinical notes describing the patient's current health status. \
    Each note is separated by a header with the date that the note was written, as well as a long list of asterisks.

    {'-' * 100}

    {notes}

    {'-' * 100}

    # Current Date
    Assume that the current date is: {current_date}

    # Question
    Does the patient meet the inclusion criterion "{self.subject}"?
    """
        return prompt

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for split in ["train", "test"]:
            # limit to zero shot setting
            if split == "test":
                path_to_data = self.path_to_train_dir if split == "train" else self.path_to_test_dir
                ensure_directory_exists(path_to_data)

                # Load dataset
                dataloader = XMLDataLoader(path_to_data)
                dataset = dataloader.load_data()

                # Create instances
                for patient in dataset:
                    is_met: bool = patient["labels"][self.subject]
                    correct_answer: str = "yes" if is_met else "no"

                    # Build `References. The possible answer choices are "yes" or "no"
                    references: List[Reference] = [
                        Reference(Output(text=answer), tags=[CORRECT_TAG] if answer == correct_answer else [])
                        for answer in N2C2CTMatchingScenario.POSSIBLE_ANSWER_CHOICES
                    ]

                    instances.append(
                        Instance(
                            input=Input(text=self.create_prompt(patient)),
                            references=references,
                            split=TRAIN_SPLIT if split == "train" else TEST_SPLIT,
                        )
                    )

        return instances
