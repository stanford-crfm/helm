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

class EhrSqlScenario1(Scenario):
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
                    input_text = f"Question: {entry['question']}\n"
                    input_text += f"Database: {entry.get('db_id', 'N/A')}\n"
                    input_text += f"Template: {entry.get('template', 'N/A')}\n"
                    input_text += f"Tag: {entry.get('tag', 'N/A')}\n"
                    input_text += f"Department: {entry.get('department', 'N/A')}\n"
                    input_text += f"Importance: {entry.get('importance', 'N/A')}\n"
                    input_text += f"Type: {entry.get('para_type', 'N/A')}\n"
                    input_text += f"Is Impossible: {entry.get('is_impossible', False)}\n"
                    input_text += f"ID: {entry.get('id', 'N/A')}\n"

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


class EhrSqlScenario(Scenario):
    """
    Scenario for the EHR SQL dataset.

    - Downloads and sets up the EHR SQL dataset.
    - Downloads and ensures availability of the `eicu.sqlite` database.
    - Generates instances for text-to-SQL evaluation.
    """

    BASE_URL: str = "https://raw.githubusercontent.com/glee4810/EHRSQL/refs/heads/main/dataset/ehrsql/eicu"
    DB_URL: str = "https://raw.githubusercontent.com/raulista1997/benchmarkdata/main/eicu.sqlite"

    name = "ehr_sql"
    description = "Generate SQL queries based on medical questions."
    tags = ["sql", "medical", "reasoning"]

    def download_file(self, url: str, save_path: str):
        """Download a file from a URL if it does not already exist."""
        if not os.path.exists(save_path):
            print(f"Downloading {url} to {save_path}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Download complete: {save_path}")

    def setup_database(self, output_path: str) -> str:
        """Ensure the SQLite database is downloaded and accessible."""
        ensure_directory_exists(output_path)
        db_path = os.path.join(output_path, "eicu.sqlite")
        self.download_file(self.DB_URL, db_path)
        return db_path

    def download_json(self, split_name: str, output_path: str) -> str:
        """Download a specific JSON file for the given split (train, valid, test)."""
        json_path = os.path.join(output_path, f"ehrsql_{split_name}.json")
        json_url = f"{self.BASE_URL}/{split_name}.json"

        if not os.path.exists(json_path):
            response = requests.get(json_url)
            response.raise_for_status()
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(response.text)

        return json_path

    def process_json(self, json_path: str, split: str) -> List[Instance]:
        """Read and process the JSON file to generate HELM instances."""
        instances: List[Instance] = []
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for entry in raw_data:
                # Ensure required fields exist
                if "question" in entry and "query" in entry:
                    input_text = f"Question: {entry['question']}\n"
                    input_text += f"Template: {entry.get('template', 'N/A')}\n"
                    input_text += f"Tag: {entry.get('tag', 'N/A')}\n"
                    input_text += f"Department: {entry.get('department', 'N/A')}\n"
                    input_text += f"Importance: {entry.get('importance', 'N/A')}\n"
                    input_text += f"Type: {entry.get('para_type', 'N/A')}\n"
                    input_text += f"Is Impossible: {entry.get('is_impossible', False)}\n"
                    input_text += f"ID: {entry.get('id', 'N/A')}\n"

                    # Create an instance
                    instance = Instance(
                        input=Input(text=input_text),
                        references=[
                            Reference(
                                Output(text=entry["query"]),
                                tags=[CORRECT_TAG],
                            )
                        ],
                        extra_data={"db_path": "eicu.sqlite"},
                        split=split,
                    )
                    instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Download and process the dataset to generate HELM instances."""
        ensure_directory_exists(output_path)

        # Ensure the SQLite database is downloaded
        self.setup_database(output_path)

        # Process dataset
        splits = {"train": TRAIN_SPLIT, "valid": TEST_SPLIT, "test": TEST_SPLIT}
        instances: List[Instance] = []

        for split_name, helm_split in splits.items():
            json_path = self.download_json(split_name, output_path)
            instances.extend(self.process_json(json_path, helm_split))

        return instances
