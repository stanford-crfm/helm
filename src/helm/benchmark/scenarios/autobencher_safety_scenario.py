import json
import os
import pandas as pd
from typing import List

from helm.common.general import ensure_file_downloaded

from .scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class AutobencherSafetyScenario(Scenario):
    """
    Autobencher safety scenario
    """

    name = "autobencher_safety"
    description = "Autobencher safety"
    tags = ["safety"]

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        os.makedirs(data_path, exist_ok=True)
        url = "https://raw.githubusercontent.com/farzaank/AutoBencher/refs/heads/main/safety_processing/process%20full%20dataset%20for%20mTurk/full_dataset.json"  # noqa: E501
        outf_path = os.path.join(data_path, "harmbench_gcg.csv")
        ensure_file_downloaded(
            source_url=url,
            target_path=outf_path,
            unpack=False,
        )

        json_data = json.loads(outf_path)
        df = pd.DataFrame(json_data)

        # Read all the instances
        instances: List[Instance] = []

        for i, row in df.iterrows():
            references = [
                Reference(output=Output(text=row[column_name]), tags=[])
                for column_name in ["category", "parent_category", "gold_answer"]
            ]
            input_text = row["question"]
            input = Input(text=input_text)
            id = str(row["category"]) + str(row["id"])
            instance = Instance(input=input, split=TEST_SPLIT, references=references, id=id)
            instances.append(instance)
        return instances
