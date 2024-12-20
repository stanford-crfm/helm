import os
import random
import csv
from typing import List, Dict

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    ALL_SPLITS,
    CORRECT_TAG,
    Input,
    PassageQuestionInput,
    Output,
)


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

    Since there are multiple questions per document and we are unlikely to test every single one,
    we randomly sample one question per document.

    More concretely, we prompt models using the following format

        <story summary>
        Question: <question>
        Answer:

        Target completion:
            <answer>

    Using an example from the training dataset, we have

    ```
    Summary: Mark Hunter (Slater), a high school student in a sleepy suburb of Phoenix, Arizona,
    starts an FM pirate radio station that broadcasts from the basement of his parents' house.
    Mark is a loner, an outsider, whose only outlet for his teenage angst and aggression is his ...
    Question: Who is Mark Hunter?
    Answer:
    ```

    Target completion:

    ```
    A loner and outsider student with a radio station.
    ```

    or

    ```
    He is a high school student in Phoenix.
    ```
    """

    name = "narrativeqa"
    description = "Question answering using summaries of books/movie scripts."
    tags = ["question_answering"]

    @staticmethod
    def get_context(summary: str, question: str) -> Input:
        """
        We follow the format from https://arxiv.org/abs/2005.14165.
        For more details, see the examples in Appendix G.
        """
        if question[-1] != "?":
            question = question + "?"
        return PassageQuestionInput(passage=summary, question=question)

    def get_split_instances(self, summaries_file: str, qaps_file: str, split: str) -> List[Instance]:
        """
        Helper for generating instances for a split.
        Args:
            summaries_file (str): File path for summaries (summaries.csv)
            qaps_file (str): File path for the question answer pairs (qaps.csv)
            split (str): Split (one of "train", "valid" or "test")

        Returns:
            List[Instance]: Instances for the specified split
        """
        split_instances: List[Instance] = []
        split_summaries: Dict[str, Dict[str, str]] = {}

        with open(summaries_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["set"] != split:
                    continue
                split_summaries[row["document_id"]] = row

        doc_id_to_question_rows: Dict[str, List[Dict[str, str]]] = {}
        with open(qaps_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["set"] != split:
                    continue
                document_id: str = row["document_id"]
                if document_id in doc_id_to_question_rows:
                    doc_id_to_question_rows[document_id].append(row)
                else:
                    doc_id_to_question_rows[document_id] = [row]

        for document_id, rows in doc_id_to_question_rows.items():
            row = random.choice(rows)
            summary: str = split_summaries[document_id]["summary"]
            question: str = row["question"]
            answer1: str = row["answer1"]
            answer2: str = row["answer2"]

            instance: Instance = Instance(
                input=self.get_context(summary.strip(), question.strip()),
                references=[
                    Reference(Output(text=answer1), tags=[CORRECT_TAG]),
                    Reference(Output(text=answer2), tags=[CORRECT_TAG]),
                ],
                split=split,
            )
            split_instances.append(instance)

        return split_instances

    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        repo_url: str = "https://github.com/deepmind/narrativeqa/archive/master.zip"
        repo_path: str = os.path.join(data_path, "narrativeqa-master")

        ensure_file_downloaded(source_url=repo_url, target_path=repo_path, unpack=True)

        # We will use the summaries, and the corresponding question and answer pairs.
        summaries_file: str = os.path.join(repo_path, "third_party", "wikipedia", "summaries.csv")
        qaps_file: str = os.path.join(repo_path, "qaps.csv")

        random.seed(0)  # we randomly pick one question per document
        instances: List[Instance] = []
        for split in ALL_SPLITS:
            instances.extend(self.get_split_instances(summaries_file=summaries_file, qaps_file=qaps_file, split=split))

        return instances
