import os
from typing import List

from common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT, CORRECT_TAG


class BabiQAScenario(Scenario):
    """
    The bAbI dataset is from the paper:
        https://arxiv.org/abs/1502.05698

    Original repository can be found at:
        https://github.com/facebookarchive/bAbI-tasks

    bAbi is a QA dataset containing 20 reasoning tasks:
    1. Single supporting fact
    2. Two supporting facts
    3. Three supporting facts 12%
    4. Binary relations (the office is north of the kitchen)
    5. Ternary relations (Mary gave the cake to Bill)
    6. Yes/No Questions
    7. Counting
    8. Lists/Sets (what items is he holding?)
    9. Negation
    10. Indefinite Knowledge (maybe, could be)
    11. Basic Coreference (he, she)
    12. Conjunction (and)
    13. Compound Coreference (they)
    14. Temporal reasoning (before, after)
    15. Deduction (transitive reasoning)
    16. Induction
    17. Spatial Reasoning (right, left, on top)
    18. Size Reasoning (smaller, larger)
    19. Path finding
    20. Motivation (Why did he go to the kitchen?)

    each sample contains a passage (an ordered list of facts), a question and
    an answer that are generated in an unconstrained/unprompted setting.

    We prompt models using the following format:
        Input sequence:
            passage: <passage>
            question: <question>
            answer:

        Output Sequence (Target completion):
            <answer>

    Using an example from the training dataset, we have
            Mary moved to the bathroom.
            John went to the hallway.
            Daniel went back to the hallway.
            Sandra moved to the garden.
            John moved to the office.
            Sandra journeyed to the bathroom.
            Where is Daniel? hallway
            Mary moved to the hallway.
            Daniel travelled to the office.
            Where is Daniel?

        Target completion:
            office
    """

    name = "babi_qa"
    description = "Question answering dataset with reasoning questions."
    tags = ["question_answering"]

    def __init__(self, task):
        self.task = int(task)

    def process_path(self, path: str) -> str:
        """Turn a path string (task 19) from the original format 's,w' to a verbal model-friendly format 'south west'"""
        steps: List[str] = path.split(",")
        directions = {"s": "south", "n": "north", "e": "east", "w": "west"}
        path = " ".join([directions[step] for step in steps])
        return path

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        splits = {"train": TRAIN_SPLIT, "valid": VALID_SPLIT, "test": TEST_SPLIT}

        url: str = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"
        target_path: str = f"{data_path}/tasks_1-20_v1-2"
        ensure_file_downloaded(source_url=url, target_path=target_path, unpack=True)

        for split in splits:
            split_path: str = f"{data_path}/tasks_1-20_v1-2/en-valid/qa{self.task}_{split}.txt"
            with open(split_path, "r") as f:

                facts = list(f)
                story: List[str] = []
                for fact in facts:
                    fid = int(fact.split(" ")[0])
                    if fid == 1:
                        story = []
                    fact = " ".join(fact.split(" ")[1:])
                    is_question = "?" in fact
                    if is_question:
                        question, answer = fact.split("\t")[:2]
                        question, answer = question.strip(), answer.strip()
                        # All tasks except task 19 have a verbal single-word answer (e.g. "kitchen", "apple", "yes").
                        # Task 19 (path finding) has a non verbal answer format (
                        if self.task == 19:
                            answer = self.process_path(answer)
                        context = f"passage: {''.join(story)}question: {question}"
                        instance: Instance = Instance(
                            input=context,
                            references=[Reference(output=answer, tags=[CORRECT_TAG])],
                            split=splits[split],
                        )
                        instances.append(instance)
                    else:
                        story.append(fact)

        return instances
