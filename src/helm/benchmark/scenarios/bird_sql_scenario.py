import json
import os
from typing import Dict, List

from filelock import FileLock

from helm.common.general import ensure_directory_exists, ensure_file_downloaded, shell
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.bird_sql_scenario_helper import (  # type: ignore
    generate_comment_prompt,
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


class BIRDSQLScenario(Scenario):
    """BIRD-SQL (Dev)"""

    name = "bird_sql"
    description = "bird_sql"
    tags = ["sql"]

    COT_PROMPT = """
Think step by step, then generate a single SQL query in valid SQLite syntax. Respond with only your reasoning and SQL query in the following tag-delimited format:

<reasoning>
INSERT_YOUR_REASONING_HERE
</reasoning>
<sql>
INSERT_YOUR_SQL_QUERY_HERE
</sql>"""  # noqa: E501

    def get_instances(self, output_path: str) -> List[Instance]:
        data_root_path = os.path.join(output_path, "dev")
        ensure_file_downloaded(
            "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip", data_root_path, unpack=True, unpack_type="unzip"
        )
        databases_unzip_target = os.path.join(data_root_path, "unzipped_dev_databases")
        _ensure_file_unzipped(os.path.join(data_root_path, "dev_databases.zip"), databases_unzip_target)
        # Note: Zip file contains .DS_Store file at the root, which makes dev_databases unzip into a nested directory
        databases_root_path = os.path.join(databases_unzip_target, "dev_databases")

        database_schema_prompts: Dict[str, str] = {}
        for database_name in os.listdir(databases_root_path):
            database_path = os.path.join(databases_root_path, database_name, f"{database_name}.sqlite")
            print(database_path)
            if not os.path.exists(database_path):
                # Ignore stray ".DS_Store" directory
                continue

            database_schema_prompt = generate_schema_prompt(database_path, num_rows=None)
            database_schema_prompts[database_name] = database_schema_prompt

        instances: List[Instance] = []
        dataset_path = os.path.join(data_root_path, "dev.json")
        dataset = json.load(open(dataset_path, "r"))
        for row in dataset:
            question_id: int = row["question_id"]
            database_id: str = row["db_id"]
            question: str = row["question"]
            knowledge: str = row["evidence"]
            gold_sql: str = row["SQL"]

            schema_prompt = database_schema_prompts[database_id]
            comment_prompt = generate_comment_prompt(question, knowledge)
            combined_prompt = schema_prompt + "\n\n" + comment_prompt + self.COT_PROMPT
            instance = Instance(
                id=f"id{question_id}",
                input=Input(text=combined_prompt),
                references=[Reference(output=Output(text=gold_sql), tags=[CORRECT_TAG])],
                extra_data={"db_id": row["db_id"]},
                split=VALID_SPLIT,
            )
            instances.append(instance)
        return instances
