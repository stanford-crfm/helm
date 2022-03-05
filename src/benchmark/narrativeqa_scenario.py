import os
import csv
from typing import List, Dict

from common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, CORRECT_TAG, TEST_SPLIT


class NarrativeQAScenario(Scenario):
    """
    The NarrativeQA dataset is from the paper:
        https://arxiv.org/abs/1712.07040

    Original repository can be found at:
        https://github.com/deepmind/narrativeqa

    This scenario is adapted from https://huggingface.co/datasets/narrativeqa

    NarrativeQA is a QA dataset containing 1,567 stories (1,102 training, 115 dev, 355 test),
    and 46,765 question-answer pairs (32,747 train, 3,461 dev, 10,557 test).
    In this Scenario, we implement the summaries-only question answering setting.

    Particularly, given the summary of a long document (either a book or a movie script),
    the goal is to answer non-localized questions.
    All of the questions and answers are written by human annotators.
    For more details, see https://arxiv.org/abs/1712.07040.


    More concretely, we prompt models using the following format:
        <story summary>
        question: <question>
        answer:

        Target completion:
            <answer>

    Using an example from the training dataset, we have
    Mark Hunter (Slater), a high school student in a sleepy suburb of Phoenix, Arizona,
    starts an FM pirate radio station that broadcasts from the basement of his parents' house.
    Mark is a loner, an outsider, whose only outlet for his teenage angst and aggression is his ...
    question: Who is Mark Hunter?
    answer:

        Target completion:
            A loner and outsider student with a radio station.
            or
            He is a high school student in Phoenix.
    """

    name = "narrativeqa"
    description = "Question answering using summaries of books/movie scripts."
    tags = ["question_answering"]

    def __init__(self):
        pass

    def get_context(self, summary: str, question: str) -> str:
        """
        We follow the format from https://arxiv.org/abs/2005.14165.
        For more details, see the examples in Appendix G.
        """
        return f"{summary}\nquestion: {question}"

    def get_split_instances(self, summaries_file: str, qaps_file: str, split_name: str, split: str) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            summaries_file (str): File path for summaries (summaries.csv)
            qaps_file (str): File path for the question answer pairs (qaps.csv)
            split (str): Split (one of "train", "valid" or "test")
            tags (List[str]): Desired tags for the instances.

        Returns:
            List[Instance]: Instances for the specified split
        """
        split_instances: List[Instance] = []
        split_summaries: Dict[str, Dict[str, str]] = {}

        with open(summaries_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["set"] != split_name:
                    continue
                split_summaries[row["document_id"]] = row

        with open(qaps_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["set"] != split_name:
                    continue
                document_id: str = row["document_id"]
                summary: str = split_summaries[document_id]["summary"]
                question: str = row["question"]
                answer1: str = row["answer1"]
                answer2: str = row["answer2"]
                context: str = self.get_context(summary, question)

                instance: Instance = Instance(
                    input=context,
                    references=[
                        Reference(output=answer1, tags=[CORRECT_TAG]),
                        Reference(output=answer2, tags=[CORRECT_TAG]),
                    ],
                    split=split,
                )
                split_instances.append(instance)

        return split_instances

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        splits = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT, "test": TEST_SPLIT}

        repo_url: str = "https://github.com/deepmind/narrativeqa/archive/master.zip"
        repo_path: str = os.path.join(data_path, "narrativeqa-master")

        ensure_file_downloaded(source_url=repo_url, target_path=repo_path, unpack=True)

        # We will use the summaries, and the corresponding question and answer pairs.
        summaries_file: str = os.path.join(repo_path, "third_party", "wikipedia", "summaries.csv")
        qaps_file: str = os.path.join(repo_path, "qaps.csv")

        instances: List[Instance] = []
        for split_name, split in splits.items():
            instances.extend(
                self.get_split_instances(
                    summaries_file=summaries_file, qaps_file=qaps_file, split_name=split_name, split=split
                )
            )

        return instances
