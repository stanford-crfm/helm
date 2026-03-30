import os
from typing import List

import datasets

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)
from helm.common.general import ensure_directory_exists


class ArabicLegalScenario(Scenario):
    name = "arabic_legal"
    description = "Arabic Legal"
    tags = ["legal"]

    TASK_NAME_QA = "qa"
    TASK_NAME_RAG = "rag"

    def __init__(self, task: str):
        super().__init__()
        if task != self.TASK_NAME_QA and task != self.TASK_NAME_RAG:
            raise ValueError(
                f"Unknown task '{task}'; " f"allowed values are {self.TASK_NAME_QA} and {self.TASK_NAME_RAG}"
            )
        self.task = task

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "yifanmai/arabic-enterprise",
            "legal",
            cache_dir=cache_dir,
            split="test",
            # TODO: Set revision
        )
        assert isinstance(dataset, datasets.Dataset)

        instances: List[Instance] = []
        for row in dataset:
            if self.task == "qa":
                input = Input(text=row["question"])
            elif self.task == "rag":
                input = Input(text=f"{row['context']}\n\n{row['question']}")
            references = [
                Reference(Output(text=row["answer"]), tags=[CORRECT_TAG]),
            ]
            instances.append(
                Instance(
                    input=input,
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(name=self.name, main_metric="exact_match", main_split="test")


class ArabicLegalQAScenario(ArabicLegalScenario):
    name = "arabic_legal_qa"
    description = "Arabic Legal QA"
    tags = ["legal"]

    def __init__(self):
        super().__init__(task="qa")


class ArabicLegalRAGScenario(ArabicLegalScenario):
    name = "arabic_legal_rag"
    description = "Arabic Legal RAG"
    tags = ["legal"]

    def __init__(self):
        super().__init__(task="rag")
