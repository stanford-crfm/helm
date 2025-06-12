import os
import json
from typing import List
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
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class EhrSqlScenario(Scenario):
    """
    Scenario for the EHR SQL dataset.

    - Downloads and sets up the EHR SQL dataset.
    - Ensures the `eicu.sqlite` database is available for evaluation.
    - Extracts schema from `eicu.sql` to pass it to the LLM.
    - Includes `value` field as alternative ground truth result.
    """

    BASE_URL: str = (
        "https://raw.githubusercontent.com/glee4810/EHRSQL/e172ec8e61391ecae2d872c8d0ba02a622222f54/dataset/ehrsql/eicu"
    )
    DB_URL: str = (
        "https://github.com/raulista1997/benchmarkdata/raw/c4c252443fa9c52afb6960f53e51be278639bea2/eicu.sqlite"
    )
    SQL_SCHEMA_URL: str = (
        "https://raw.githubusercontent.com/glee4810/EHRSQL/"
        "e172ec8e61391ecae2d872c8d0ba02a622222f54/dataset/ehrsql/eicu/eicu.sql"
    )

    name = "ehr_sql"
    description = "Given a natural language instruction, generate an SQL query that would be used in clinical research."
    tags = ["sql", "medical", "reasoning"]

    def setup_database(self, output_path: str) -> str:
        """Ensure the SQLite database is downloaded and accessible."""
        ensure_directory_exists(output_path)
        db_path = os.path.join(output_path, "eicu.sqlite")
        ensure_file_downloaded(self.DB_URL, db_path)
        return db_path

    def download_sql_schema(self, output_path: str) -> str:
        """Download the SQL schema file containing table definitions."""
        ensure_directory_exists(output_path)
        schema_path = os.path.join(output_path, "eicu.sql")
        ensure_file_downloaded(self.SQL_SCHEMA_URL, schema_path)
        return schema_path

    def extract_schema(self, schema_path: str) -> str:
        """Extracts and returns `CREATE TABLE` statements from the SQL schema file."""
        schema_text = []
        with open(schema_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        inside_create_table = False
        current_table_def = []

        for line in lines:
            line = line.strip()

            if line.startswith("CREATE TABLE"):
                inside_create_table = True
                current_table_def = [line]
            elif inside_create_table:
                current_table_def.append(line)
                if line.endswith(";"):  # End of table definition
                    inside_create_table = False
                    schema_text.append("\n".join(current_table_def))

        return "\n\n".join(schema_text)

    def download_json(self, split_name: str, output_path: str) -> str:
        """Download a specific JSON file for the given split (train, valid, test)."""
        json_path = os.path.join(output_path, f"ehrsql_{split_name}.json")
        json_url = f"{self.BASE_URL}/{split_name}.json"
        ensure_file_downloaded(json_url, json_path)
        return json_path

    def process_json(self, json_path: str, schema_prompt: str, split: str) -> List[Instance]:
        """Read and process the JSON file to generate HELM instances."""
        instances: List[Instance] = []
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            for entry in raw_data:
                if "question" in entry and "query" in entry:
                    is_impossible = entry.get("is_impossible", False)
                    input_text = f"-- Database Schema:\n{schema_prompt}\n\n-- \
                    Question:\n{entry['question']}\n\n"

                    instance = Instance(
                        input=Input(text=input_text),
                        references=[Reference(Output(text=entry["query"]), tags=[CORRECT_TAG])],
                        extra_data={
                            "db_path": "eicu.sqlite",
                            "value": entry.get("value", {}),
                            "is_impossible": is_impossible,
                        },
                        split=split,
                    )

                    instances.append(instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Download and process the dataset to generate HELM instances."""
        ensure_directory_exists(output_path)

        # Ensure SQLite database is downloaded
        self.setup_database(output_path)

        # Ensure schema file is downloaded and extract schema
        schema_path = self.download_sql_schema(output_path)
        schema_prompt = self.extract_schema(schema_path)

        # Process dataset
        splits = {"train": TRAIN_SPLIT, "valid": TEST_SPLIT, "test": TEST_SPLIT}
        instances: List[Instance] = []

        for split_name, helm_split in splits.items():
            json_path = self.download_json(split_name, output_path)
            instances.extend(self.process_json(json_path, schema_prompt, helm_split))

        return instances
