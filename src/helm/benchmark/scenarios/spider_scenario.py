import json
import os
from typing import Dict, List

from filelock import FileLock

from helm.common.general import ensure_directory_exists, ensure_file_downloaded, shell
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.bird_sql_scenario_helper import (  # type: ignore
    generate_schema_prompt,
)
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Scenario,
    Instance,
    Reference,
    VALID_SPLIT,
    Input,
    Output,
)


def _ensure_file_unzipped(source_path: str, target_path: str):
    with FileLock(f"{target_path}.lock"):
        if os.path.exists(target_path):
            hlog(f"Not decompressing {source_path} because {target_path} already exists")
            return
        tmp_path = target_path + ".tmp"
        ensure_directory_exists(tmp_path)
        shell(["unzip", source_path, "-d", tmp_path])
        shell(["mv", tmp_path, target_path])


class SpiderScenario(Scenario):
    """Spider 1.0"""

    name = "spider"
    description = "spider"
    tags = ["sql"]

    INSTRUCTIONS_PROMPT = """-- Using valid SQLite, answer the following questions for the tables provided above.
"""
    COT_PROMPT = """
Think step by step, then generate a single SQL query in valid SQLite syntax. Respond with only your reasoning and SQL query in the following tag-delimited format:

<reasoning>
INSERT_YOUR_REASONING_HERE
</reasoning>
<sql>
INSERT_YOUR_SQL_QUERY_HERE
</sql>"""  # noqa: E501

    def get_instances(self, output_path: str) -> List[Instance]:
        data_parent_path = os.path.join(output_path, "data")
        ensure_file_downloaded(
            "https://drive.google.com/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t",
            data_parent_path,
            unpack=True,
            unpack_type="unzip",
        )
        data_root_path = os.path.join(data_parent_path, "spider_data")
        databases_root_path = os.path.join(data_root_path, "test_database")

        database_schema_prompts: Dict[str, str] = {}
        for database_name in os.listdir(databases_root_path):
            database_path = os.path.join(databases_root_path, database_name, f"{database_name}.sqlite")
            if not os.path.exists(database_path):
                # Ignore stray ".DS_Store" directory
                continue

            database_schema_prompt = generate_schema_prompt(database_path, num_rows=None)
            database_schema_prompts[database_name] = database_schema_prompt

        instances: List[Instance] = []
        dataset_path = os.path.join(data_root_path, "test.json")
        dataset = json.load(open(dataset_path, "r"))
        for row in dataset:
            database_id: str = row["db_id"]
            question: str = row["question"]
            gold_sql: str = row["query"]

            schema_prompt = database_schema_prompts[database_id]
            combined_prompt = schema_prompt + "\n\n" + self.INSTRUCTIONS_PROMPT + question + self.COT_PROMPT
            instance = Instance(
                input=Input(text=combined_prompt),
                references=[Reference(output=Output(text=gold_sql), tags=[CORRECT_TAG])],
                extra_data={"db_id": row["db_id"]},
                split=VALID_SPLIT,
            )
            instances.append(instance)
        return instances
