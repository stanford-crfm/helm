import os
import json
from typing import List, Dict
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_directory_exists
import requests

class EhrSqlScenario(Scenario):
    """
    Scenario for the EHR SQL dataset.

    Dataset Overview:
    EHR SQL is a dataset designed for text-to-SQL tasks in the context of electronic health records (EHRs).
    It consists of utterances collected from 222 hospital staff members, including physicians, nurses, and
    health records teams. The dataset is linked to two open-source EHR databases, MIMIC-III and eICU.

    Key Features:
    - Questions reflect real-world hospital needs, ranging from simple retrieval to complex calculations.
    - Includes time-sensitive queries and unanswerable questions for robust evaluation.
    - Offers a benchmark for developing and assessing text-to-SQL models in healthcare.

    Challenges:
    - Understanding various time-sensitive medical expressions in the questions.
    - Generating complex SQL queries for calculations like survival rates.
    - Distinguishing between answerable and unanswerable questions.

    Dataset Source:
    The EHR SQL dataset is available at:
    https://github.com/glee4810/EHRSQL
    """

    BASE_URL: str = "https://raw.githubusercontent.com/glee4810/EHRSQL/refs/heads/main/dataset/ehrsql/eicu"

    name = "ehr_sql"
    description = "Generate SQL queries based on medical questions."
    tags = ["sql", "medical", "reasoning"]

    def download_json(self, split_name: str, output_path: str) -> str:
        """
        Download a specific JSON file for the given split (train, valid, test).
        """
        json_path = os.path.join(output_path, f"ehrsql_{split_name}.json")
        json_url = f"{self.BASE_URL}/{split_name}.json"

        if not os.path.exists(json_path):
            response = requests.get(json_url)
            response.raise_for_status()
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(response.text)

        return json_path

    def process_json(self, json_path: str, split: str) -> List[Instance]:
        """
        Read and process the JSON file to generate HELM instances.
        """
        instances: List[Instance] = []
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for entry in raw_data:
                # Ensure the entry has required fields
                if "question" in entry and "query" in entry and "value" in entry:
                    input_text = f"Question: {entry['question']}"

                    # Create an instance
                    instance = Instance(
                        input=Input(text=input_text),
                        references=[
                            Reference(
                                Output(text=entry["query"]),
                                tags=[CORRECT_TAG],
                            )
                        ],
                        split=split,
                    )
                    instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Download and process the dataset to generate HELM instances.
        - Train data is processed into the TRAIN_SPLIT.
        - Validation and test data are merged into the TEST_SPLIT.
        """
        ensure_directory_exists(output_path)
        splits = {"train": TRAIN_SPLIT, "valid": TEST_SPLIT, "test": TEST_SPLIT}
        instances: List[Instance] = []

        for split_name, helm_split in splits.items():
            json_path = self.download_json(split_name, output_path)
            instances.extend(self.process_json(json_path, helm_split))

        return instances
