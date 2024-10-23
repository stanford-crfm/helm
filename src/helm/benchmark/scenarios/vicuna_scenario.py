import json
import os
from typing import List

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class VicunaScenario(Scenario):
    """
    This scenario is based on the questions used by the Vicuna team to evaluate instruction-following models.

    https://lmsys.org/blog/2023-03-30-vicuna/
    """

    name = "vicuna"
    description = "Vicuna eval questions"
    tags = ["instructions"]

    def __init__(self, category: str):
        super().__init__()
        self.category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        def matches_target_category(raw: dict) -> bool:
            return self.category == "all" or raw["category"] == self.category

        # Download the raw data
        source_url = "https://raw.githubusercontent.com/lm-sys/FastChat/v0.2.5/fastchat/eval/table/question.jsonl"
        data_path: str = os.path.join(output_path, "vicuna_questions.jsonl")

        ensure_file_downloaded(
            source_url=source_url,
            target_path=data_path,
        )

        # Read all the instances
        instances: List[Instance] = []
        for line in open(data_path):
            # Example: {"question_id": 1, "text": "How can I improve my time management skills?", "category": "generic"}
            raw = json.loads(line)
            if matches_target_category(raw):
                instance = Instance(
                    input=Input(text=raw["text"]),
                    split=TEST_SPLIT,
                    references=[],
                )
                instances.append(instance)
        return instances
