import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Output,
    Reference,
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

    def __init__(self, subset: str):
        super().__init__()
        self.subset = subset

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset("yifanmai/call-center", self.subset, split="test", cache_dir=cache_dir)
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["transcript"])
            instance = Instance(input=input, references=[], split=TEST_SPLIT)
            instances.append(instance)
        return instances


class CallCenterSummarizationPairwiseComparisonScenario(Scenario):
    """Call center summarization."""

    name = "call_center_summarization_pairwise_comparison"
    description = "Call center summarization."
    tags = ["call_center"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "yifanmai/call-center", "summarization_with_annotations", split="test", cache_dir=cache_dir
        )
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["transcript"])
            reference = Reference(output=Output(text=row["gpt-4o-mini-2024-07-18_summary"]), tags=[CORRECT_TAG])
            instance = Instance(input=input, references=[reference], split=TEST_SPLIT)
            instances.append(instance)
        return instances


class CallCenterSummarizationKeyPointsRecallScenario(Scenario):
    """Call center summarization."""

    name = "call_center_summarization_key_points_recall"
    description = "Call center summarization."
    tags = ["call_center"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        dataset = datasets.load_dataset(
            "yifanmai/call-center", "summarization_with_annotations", split="test", cache_dir=cache_dir
        )
        instances: List[Instance] = []
        for row in dataset:
            input = Input(text=row["transcript"])
            references = [
                Reference(output=Output(text=key_point), tags=[CORRECT_TAG])
                for key_point in row["gpt-4o-mini-2024-07-18_key_points"]
            ]
            instance = Instance(input=input, references=references, split=TEST_SPLIT)
            instances.append(instance)
        return instances
