import os
from typing import List

from helm.common.codec import from_jsonl
from helm.benchmark.scenarios.scenario import Scenario, Instance


class CustomMCQAScenario(Scenario):
    """
    We prompt models using the following format

        <input>                  # train
        A. <reference>
        B. <reference>
        C. <reference>
        D. <reference>
        Answer: <A/B/C/D>

        x N (N-shot)

        <input>                  # test
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer:

    For example (from mmlu:anatomy), we have:

        The pleura
        A. have no sensory innervation.
        B. are separated by a 2 mm space.
        C. extend into the neck.
        D. are composed of respiratory epithelium.
        Answer: C

        Which of the following terms describes the body's ability to maintain its normal state?
        A. Anabolism
        B. Catabolism
        C. Tolerance
        D. Homeostasis
        Answer:

    Target: D
    """

    name = "custom_mcqa"
    description = "A generic multiple choice question answering scenario that a user "
    "can use to benchmark a model on their own data."
    tags = ["multiple_choice"]

    def __init__(
        self,
        path: str,
        num_train_instances: int = 0,
    ):
        super().__init__()
        self.path: str = path
        self.num_train_instances: int = num_train_instances

    def process_jsonl(self, jsonl_path: str) -> List[Instance]:
        instances: List[Instance] = []
        with open(jsonl_path, mode="r") as reader:
            instances = from_jsonl(reader.read(), Instance)

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        # Read all the instances
        path: str = os.path.join(output_path, self.path)
        instances: List[Instance] = self.process_jsonl(path)
        return instances
