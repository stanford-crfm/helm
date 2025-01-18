import csv
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_file_downloaded


class MedecScenario(Scenario):
    """
    Processes the MEDEC dataset for medical error detection and correction tasks.

    MEDEC is the first publicly available benchmark for medical error detection and correction in clinical notes,
    introduced in "Ben Abacha et al., 2024." The dataset includes 3,848 clinical texts from the MS and UW collections,
    covering five types of errors:
    - Diagnosis
    - Management
    - Treatment
    - Pharmacotherapy
    - Causal Organism

    The dataset consists of:
    - Training Set: 2,189 MS texts
    - Validation Set: 574 MS texts and 160 UW texts
    - Test Set: 597 MS texts and 328 UW texts

    Each clinical text is labeled as either correct or containing one error. The task involves:
    (A) Predicting the error flag (1: the text contains an error, 0: the text has no errors).
    (B) For flagged texts, extracting the sentence that contains the error.
    (C) Generating a corrected sentence.

    The MEDEC dataset was used for the MEDIQA-CORR shared task to evaluate seventeen participating systems.
    Recent LLMs (e.g., GPT-4, Claude 3.5 Sonnet, Gemini 2.0 Flash) have been evaluated on this dataset, showing good
    performance but still lagging behind medical doctors in error detection and correction tasks.

    Task:
    Given a clinical text, models must identify errors and correct them while demonstrating medical knowledge
    and reasoning capabilities.
    """

    TRAIN_URL = "https://raw.githubusercontent.com/abachaa/MEDEC/refs/heads/main/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
    TEST_URL = "https://raw.githubusercontent.com/abachaa/MEDEC/refs/heads/main/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv"

    name = "medec"
    description = "MEDEC dataset for medical error detection and correction tasks."
    tags = ["error_detection", "error_correction", "biomedical"]

    def download_csv(self, url: str, output_path: str, file_name: str) -> str:
        """Download the CSV file and return its path."""
        csv_path = os.path.join(output_path, file_name)
        ensure_file_downloaded(source_url=url, target_path=csv_path, unpack=False)
        return csv_path

    def process_csv(self, csv_path: str, split: str) -> List[Instance]:
        """Read and process a CSV file to generate instances."""
        instances: List[Instance] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Validate required fields
                if not row.get("Text") or not row.get("Error Type"):
                    continue

                # Build input text (entire clinical note)
                input_text = row["Text"].strip()

                # Parse error-related fields
                error_flag = int(row.get("Error Flag", 0))
                error_type = row["Error Type"].strip()
                error_sentence_id = int(row.get("Error Sentence ID", -1))
                corrected_sentence = row.get("Corrected Sentence", "").strip()

                # Create references for correction task
                references = []
                if error_flag == 1 and error_sentence_id != -1:
                    references.append(
                        Reference(
                            Output(text=corrected_sentence),
                            tags=[CORRECT_TAG],
                        )
                    )

                # Create an instance for the error detection and correction task
                instance = Instance(
                    input=Input(text=input_text),
                    references=references,
                    split=split,
                )
                instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Download and process the dataset to generate instances."""
        instances: List[Instance] = []

        # Download and process the training set
        train_csv = self.download_csv(self.TRAIN_URL, output_path, "medec_train.csv")
        instances.extend(self.process_csv(train_csv, TRAIN_SPLIT))

        # Download and process the test set
        test_csv = self.download_csv(self.TEST_URL, output_path, "medec_test.csv")
        instances.extend(self.process_csv(test_csv, TEST_SPLIT))

        return instances
