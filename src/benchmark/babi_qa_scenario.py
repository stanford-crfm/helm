import os
from typing import List

from common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


class BabiQAScenario(Scenario):
    """
    The bAbI dataset is from the paper:
        https://arxiv.org/abs/1502.05698

    Original repository can be found at:
        https://github.com/facebookarchive/bAbI-tasks

    bAbi is a QA dataset containing 20 reasoning tasks,
    each sample contains a passage (an ordered list of facts), a question and
    an answer that are generated in an unconstrained/unprompted setting.

    We prompt models using the following format:
        <passage>
        question: <question>
        answer:

        Target completion:
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
        self.task = task
        pass

    def get_instances(self) -> List[Instance]:
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []
        splits = {"train": TRAIN_TAG, "valid": VALID_TAG, "test": TEST_TAG}

        url = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"
        target_path = f"{data_path}/tasks_1-20_v1-2.tar.gz"
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
                        question, asnwer = fact.split("\t")[:2]
                        question, answer = question.strip(), asnwer.strip()
                        context = f"{''.join(story)}{question}"
                        instance: Instance = Instance(
                            input=context,
                            references=[Reference(output=answer, tags=[CORRECT_TAG])],
                            tags=[splits[split]],
                        )
                        instances.append(instance)
                    else:
                        story.append(fact)

        return instances
