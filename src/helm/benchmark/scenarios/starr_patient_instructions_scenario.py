import os
import csv
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


class StarrPatientInstructionsScenario(Scenario):
    """
    Starr Patient Instructions is a dataset created from STARR-OMOP data, containing after-visit instructions
    for outpatient surgeries/procedures. Each example corresponds to one surgery or procedure case (only including
    outpatient or observation/overnight cases with discharge within 24 hours) and includes the following fields:

      - Diagnosis: Why the patient needs the surgery/procedure.
      - ActualProcedure: The surgery/procedure name.
      - HistoryPhysicalNoteText: The History & Physical note written by the surgeon.
      - OperativeNoteText: The report describing what was done during the surgery/procedure.
      - DischargeInstructionNoteText: The specific after-surgery care instructions given to the patient.

    The task is to generate personalized post-procedure patient instructions based on the provided case details.

    Sample Synthetic Prompt:
        Given the following case details, generate personalized after-surgery care instructions.

        Diagnosis: [diagnosis text]
        Procedure: [actual procedure text]
        History & Physical: [H&P note text]
        Operative Report: [operative note text]

        Patient Instructions:
    """

    name = "starr_patient_instructions"
    description = "A dataset containing case details used to generate customized post-procedure patient instructions."
    tags = ["patient_communication", "healthcare", "instruction_generation", "surgery"]

    def get_instances(self, output_path: str) -> List[Instance]:
        csv_path = "/share/pi/nigam/suhana/medhelm/data/starr-omop-personalized-care-instr/dataset_cases_qc.csv"
        # Ensure the directory for the CSV file exists.
        ensure_directory_exists(os.path.dirname(csv_path))

        instances: List[Instance] = []
        # For now, we assign all instances to the test split (zero-shot setting).
        split = TEST_SPLIT

        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Retrieve and strip the relevant fields.
                qc_value = row.get("QC", "").strip().upper()
                if qc_value != "TRUE":
                    continue
                diagnosis = row.get("Diagnosis", "").strip()
                actual_procedure = row.get("ActualProcedure", "").strip()
                history_physical = row.get("HistoryPhysicalNoteText", "").strip()
                operative_note = row.get("OperativeNoteText", "").strip()
                discharge_instruction = row.get("DischargeInstructionNoteText", "").strip()

                # Skip the instance if any required field is missing.
                if not (
                    diagnosis and actual_procedure and history_physical and operative_note and discharge_instruction
                ):
                    continue

                # Construct the input prompt by concatenating the fields.
                input_text = (
                    f"Diagnosis: {diagnosis}\n"
                    f"Procedure: {actual_procedure}\n"
                    f"History & Physical: {history_physical}\n"
                    f"Operative Report: {operative_note}\n\n"
                )

                instances.append(
                    Instance(
                        input=Input(text=input_text),
                        references=[Reference(Output(text=discharge_instruction), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
