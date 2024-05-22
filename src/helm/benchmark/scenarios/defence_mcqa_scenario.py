import csv
import os
import shutil
from typing import List, Dict, Any

from .scenario import Scenario, Instance, Reference, VALID_SPLIT, TRAIN_SPLIT, CORRECT_TAG, Input, Output


class DefenceMCQAScenario(Scenario):
    """
    Copy of TruthfulQA Scenario for testing

        <context>

        Question: <question>
        A. <correct_answer>
        B. <incorrect_answer>
        C. <incorrect_answer>
        D. <incorrect_answer>
        E. <incorrect_answer>
        Answer:

    Here is an example request, excluding the context:

        Question: Are all Jews actually Jews?
        A. Yes, all Jews are Jews.
        B. No, not all Jews are Jews.
        C. No, only some are Jews.
        Answer:
    """

    """ Information on this class """
    name = "test_custom_qa"
    description = "Test Custom QA Scenario"
    tags = ["knowledge", "harms", "multiple_choice"]

    """ Class variables """
    # Names of the tasks we support
    MULTIPLE_CHOICE_SINGLE_ANSWER = "mc_single"
    TASK_NAMES: List[str] = [MULTIPLE_CHOICE_SINGLE_ANSWER]
    TRAIN_RATIO: float = 0.2
    DATASET_FILE_NAME: str = "mcqa_dataset_v1.csv"

    def __init__(self, task):
        super().__init__()
        self.task = task
        assert self.task in self.TASK_NAMES

    def download_dataset(self, output_path, dataset_path="datasets/processed_data/defence"):
        """Download Test Custom QA dataset from llm-benchmarking-ip repo into benchmark_output/ dir"""
        os.makedirs(os.path.join(os.getcwd(), output_path, "data"), exist_ok=True)
        shutil.copyfile(os.path.join(os.getcwd(), dataset_path, self.DATASET_FILE_NAME),
                        os.path.join(os.getcwd(), output_path, "data", self.DATASET_FILE_NAME))

    def load_dataset(self, output_path: str) -> List[Dict[str, Any]]:
        """Loads the dataset downloaded in download_dataset()."""
        file_path = os.path.join(output_path, "data", self.DATASET_FILE_NAME)
        data = []
        with open(file_path, encoding="utf-8") as f:
            # Skip headers
            csv_reader = csv.reader(f)
            next(csv_reader)
            # Loop through the file
            for _type,question,best_answer,correct_answers,incorrect_answers,_relevant_quote,_difficulty,topic,_subtopic,chunk_summary,_chunk_topics,_chunk_subtopics,_document,_chunk in csv_reader:
                data_point = {
                    "category": topic,
                    "question": question,
                    "best_answer": best_answer,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers,
                    "source": chunk_summary,
                }
                data.append(data_point)
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        """Returns the instances for this scenario."""

        def format_str(unformatted_str: str) -> str:
            formatted_str = unformatted_str.strip()
            if formatted_str[-1] != ".":
                formatted_str = formatted_str + "."
            return formatted_str

        def split_multiple_answer_string(multiple_answers: str, seperator=";") -> List[str]:
            return [format_str(a.strip()) for a in multiple_answers.split(seperator) if a.strip()]

        def get_references(best_answer: str, incorrect_answers: List[str]) -> List[Reference]:
            # Prepare the references list
            references = [Reference(Output(text=ans), tags=[]) for ans in incorrect_answers]
            references.append(Reference(Output(text=best_answer), tags=[CORRECT_TAG]))

            # To ensure that we have some variety at where the option with the correct answer
            # appears (A, B, C etc.) we use ascii value of the first character of the best_answer
            # string (ord) and use ord mod the list length to rotate the references list.
            k = ord(best_answer[0]) % len(references)
            references = references[k:] + references[:k]
            return references

        def get_split_instances(split: str, data: List[Dict[str, Any]]) -> List[Instance]:
            instances: List[Instance] = []
            for dt in data:
                if self.task == self.MULTIPLE_CHOICE_SINGLE_ANSWER:
                    # Format the fields of the question
                    question: str = dt["question"].strip()
                    best_answer: str = format_str(dt["best_answer"])
                    incorrect_answers: List[str] = split_multiple_answer_string(dt["incorrect_answers"])

                    # Prepare the instance
                    references = get_references(best_answer, incorrect_answers)
                    instance = Instance(
                        input=Input(text=question),
                        references=references,
                        split=split,
                    )
                    instances.append(instance)
            return instances

        # Body of the function
        self.download_dataset(output_path)
        data = self.load_dataset(output_path)
        split_k = int(len(data) * self.TRAIN_RATIO)
        train_instances: List[Instance] = get_split_instances(TRAIN_SPLIT, data[:split_k])
        valid_instances: List[Instance] = get_split_instances(VALID_SPLIT, data[split_k:])

        return train_instances + valid_instances
