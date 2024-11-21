import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog


class AutoBencherCapabilitiesScenario(Scenario):
    """AutoBencher Capabilities

    AutoBencher uses a language model to automatically search
    for datasets. AutoBencher Capabilities consists of question
    answering datasets for math, multilingual, and knowledge-intensive
    question answering created by AutoBencher.

    Paper: https://arxiv.org/abs/2407.08351"""

    name = "autobencher_capabilities"
    description = (
        "AutoBencher Capabilities consists of question answering datasets "
        "for math, multilingual, and knowledge-intensive "
        "question answering created by AutoBencher. "
        "([paper](https://arxiv.org/abs/2407.08351))"
    )
    tags = ["question answering"]

    SUBJECTS = ["math", "mt", "econ", "science", "history"]

    def __init__(self, subject: str):
        super().__init__()
        if subject not in self.SUBJECTS:
            raise ValueError(f"Unexpected subject {subject}, available subjects are {self.SUBJECTS}")
        self.subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)

        # TODO: Switch this to the production dataset when available.
        dataset = datasets.load_dataset(
            "xlisali1/AutoBencher-capability.json",
            split="train",  # Use train split as test, so only zero-shot is supported
            cache_dir=cache_dir,
            revision="efe58dd72b6423e3f5c967f16cbea8cce3a51933",
        )
        instances: List[Instance] = []
        for row in dataset:
            if row["subject"] == self.subject:
                continue
            input = Input(text=row["question"])
            # References are category ID, followed by level 2, 3 and 4 category names.
            references = [Reference(output=Output(text=row["gold_answer"]), tags=[CORRECT_TAG])]
            if row["gold_answer"] is None:
                hlog(f"WARNING: Row had no gold_answer: {row}")
                continue
            instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)
        return instances
