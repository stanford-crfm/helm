import csv
import os
import random
from typing import List, Dict, Any
import pandas as pd

from .narrativeqa_scenario import NarrativeQAScenario
from .scenario import (
    Instance,
    Reference,
    Output
)

CORRECT_TAG: str = "correct"


class DefenceQAScenario(NarrativeQAScenario):
    name = "defenceqa"
    description = "Question answering using defence data"

    def __init__(self):
        super(DefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str, dataset_dir: str="dataset/defence_qa/") -> List[Instance]:
        base_dir = os.path.join(os.getcwd(), dataset_dir)
        df_summaries = pd.read_csv(os.path.join(base_dir, "context.csv"))
        df_qa = pd.read_csv(os.path.join(base_dir, "qa.csv"))

        instances: List[Instance] = []

        for _, row in df_qa.iterrows():
            question = row["question"]
            answer = row["expected_answer"]
            context_id = row["context_id"]
            print("context_id={}".format(context_id))
            summary = (
                df_summaries[df_summaries["context_id"].apply(lambda x: x in context_id)].iloc[0]["text_chunk"]
            )
            instance: Instance = Instance(
                input=self.get_context(summary.strip(), question.strip()),
                references=[
                    Reference(Output(text=answer), tags=[CORRECT_TAG]),
                ],
                split="test",
            )
            instances.append(instance)

        return instances


class AnalyticalDefenceQAScenario(DefenceQAScenario):
    def __init__(self):
        super(AnalyticalDefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        return super().get_instances(output_path, "datasets/defence_qa/analytical")


class OpenEndedDefenceQAScenario(DefenceQAScenario):
    def __init__(self):
        super(OpenEndedDefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        return super().get_instances(output_path, "datasets/defence_qa/open_ended")


class FactualDefenceQAScenario(DefenceQAScenario):
    def __init__(self):
        super(FactualDefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        return super().get_instances(output_path, "datasets/defence_qa/factual")
