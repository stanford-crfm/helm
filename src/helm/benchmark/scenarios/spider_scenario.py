import json
import os
from typing import Dict, List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_file_downloaded
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
    ScenarioMetadata,
)


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

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="spider",
            display_name="Spider 1.0 (Test)",
            description="Spider 1.0 (Test)",
            taxonomy=TaxonomyInfo(
                task="text-to-SQL",
                what="databases from various domains",
                when="?",
                who="expert data scientists",
                language="English",
            ),
            main_metric="execution_accuracy",
            main_split="valid",
        )
