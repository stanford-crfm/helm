import json
import os
import pandas as pd
import datasets
from typing import List

from helm.common.general import ensure_file_downloaded

from .scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class AutobencherCapabilityScenario(Scenario):
    """
    Autobencher capability scenario
    """

    name = "autobencher_capability"
    description = "Autobencher capability"
    tags = ["capability", "autobencher"]

    def get_instances(self, output_path: str) -> List[Instance]:
        huggingface_dataset = datasets.load_dataset("xlisali1/AutoBencher-capability.json")['train']

        json_data = huggingface_dataset.to_dict()
        df = pd.DataFrame(json_data)
        print(df.columns)

        # Read all the instances
        instances: List[Instance] = []

        for i, row in df.iterrows():
            references = [
                Reference(output=Output(text=row[column_name]), tags=[])
                for column_name in ["category", "parent_category", "gold_answer"]
            ]
            input_text = row["question"]
            input = Input(text=input_text)
            id = str(row["subject"]) + str(row["category"])
            instance = Instance(input=input, split=TEST_SPLIT, references=references, id=id)
            instances.append(instance)
        return instances