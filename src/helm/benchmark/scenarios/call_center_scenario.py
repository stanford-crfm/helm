import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    TEST_SPLIT,
    Input,
)
from helm.common.general import ensure_directory_exists


class CallCenterSummarizationScenario(Scenario):
    """Call center summarization."""

    name = "call_center_summarization"
    description = "Call center summarization."
    tags = ["call_center"]

    def __init__(self, revision: str):
        super().__init__()
        self.revision = revision

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset("yifanmai/call-center", "summarization", split="test", cache_dir=cache_dir)
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["transcript"])
            instance = Instance(input=input, references=[], split=TEST_SPLIT)
            instances.append(instance)
        return instances
