import csv
import os
import random
from typing import List, Dict, Any
import pandas as pd

from .narrativeqa_scenario import NarrativeQAScenario
from .scenario import (
    Instance,
    Reference,
    Output,
    PassageQuestionInput
)

CORRECT_TAG: str = "correct"


class DefenceQAScenario(NarrativeQAScenario):
    name = "defenceqa"
    description = "Question answering using defence data"
    version = "v0"

    def __init__(self):
        super(DefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str, dataset_dir: str="datasets/processed_data/defence") -> List[Instance]:
        base_dir = os.path.join(os.getcwd(), dataset_dir)
        df_summaries = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(base_dir)), "text_chunks.csv"))
        df_qa = pd.read_csv(os.path.join(base_dir, "qa.csv"))

        instances: List[Instance] = []

        for _, row in df_qa.iterrows():
            question = row["question"]
            answer = row["answer"]
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
        return super().get_instances(output_path, f"datasets/processed_data/defence/{self.version}/analytical")


class OpenEndedDefenceQAScenario(DefenceQAScenario):
    def __init__(self):
        super(OpenEndedDefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        return super().get_instances(output_path, f"datasets/processed_data/defence/{self.version}/open_ended")


class FactualDefenceQAScenario(DefenceQAScenario):
    def __init__(self):
        super(FactualDefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str) -> List[Instance]:
        return super().get_instances(output_path, f"datasets/processed_data/defence/{self.version}/factual")


class MCDefenceQAScenario(DefenceQAScenario):
    def __init__(self):
        super(MCDefenceQAScenario, self).__init__()

    def get_instances(self, output_path: str, dataset_dir=None) -> List[Instance]:
        
        if dataset_dir is None:
            dataset_dir = f"datasets/processed_data/defence/{self.version}/mcqa"

        def format_str(unformatted_str: str) -> str:
            formatted_str = unformatted_str.strip()
            if formatted_str[-1] != ".":
                formatted_str = formatted_str + "."
            return formatted_str

        def get_references(best_answer: str, incorrect_answers: List[str]) -> List[Reference]:
            references = [Reference(Output(text=ans), tags=[]) for ans in incorrect_answers]
            references.append(Reference(Output(text=best_answer), tags=[CORRECT_TAG]))
            # To ensure that we have some variety at where the option with the correct answer
            # appears (A, B, C etc.) we use ascii value of the first character of the best_answer
            # string (ord) and use ord mod the list length to rotate the references list.
            k = ord(best_answer[0]) % len(references)
            references = references[k:] + references[:k]
            return references

        base_dir = os.path.join(os.getcwd(), dataset_dir)
        df_summaries = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(base_dir)), "text_chunks.csv"))
        df_qa = pd.read_csv(os.path.join(base_dir, "qa.csv"))

        instances: List[Instance] = []

        for _, row in df_qa.iterrows():
            question: str = row["question"].strip()
            best_answer: str = format_str(row["best_answer"])
            context_id = row["context_id"]
            context = df_summaries[df_summaries["context_id"] == context_id].iloc[0]["text_chunk"]
            incorrect_answers: List[str] = [format_str(a) for a in eval(row["incorrect_answers"])]

            # Prepare the instance
            references = get_references(best_answer, incorrect_answers)
            instance = Instance(
                input=PassageQuestionInput(passage=context, question=question),
                references=references,
                split="test",
            )
            instances.append(instance)
        
        return instances
