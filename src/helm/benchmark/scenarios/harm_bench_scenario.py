import os
import re
import pandas as pd
from typing import List, Any, Dict
from datasets import load_dataset

from helm.common.general import ensure_file_downloaded

from .scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output


class HarmBenchScenario(Scenario):
    """
    HarmBench, a standardized evaluation framework for automated red teaming.
    """

    name = "harm_bench"
    description = "HarmBench dataset - standard and contextual attacks"
    tags = ["safety"]

    def __init__(self, subject: str = "all"):
        super().__init__()
        self.subset = subject

    def get_instances(self, output_path: str) -> List[Instance]:

        data_path = os.path.join(output_path, "data")
        os.makedirs(data_path, exist_ok=True)
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
            target_path=data_path,
            unpack=False,
        )

        df = pd.read_csv(data_path + "/harmbench_behaviors_text_all.csv")

        # Read all the instances
        instances: List[Instance] = []
        for i, row in df.iterrows():
            if row["FunctionalCategory"] == "contextual":
                context = row["ContextString"]
                input_text = row["Behavior"] + " You may use the following context: " + context
            else:
                input_text = row["Behavior"]
            input = Input(text=input_text)
            tag = row["Tags"]
            id = row["BehaviorID"]
            references = [Reference(output=Output(text=row["SemanticCategory"]), tags=[])]
            if tag != "":
                instance = Instance(input=input, split=TEST_SPLIT, references=references, id=id)
            else:
                instance = Instance(input=input, split=TEST_SPLIT, references=references, sub_split=tag, id=id)
            instances.append(instance)
        return instances
