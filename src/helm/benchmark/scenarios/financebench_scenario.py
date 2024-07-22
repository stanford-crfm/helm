import dataclasses
import json
import os
import random
from typing import List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TRAIN_SPLIT,
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class FinanceBenchScenario(Scenario):
    """FinanceBench"""

    name = "financebench"
    description = "FinanceBench"
    tags = ["finance"]

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        target_path = os.path.join(cache_dir, "financebench_open_source.jsonl")
        url: str = (
            "https://raw.githubusercontent.com/patronus-ai/financebench/d7beebe5e739e0b806ab4443c1b3e23f51804acf/data/financebench_open_source.jsonl"  # noqa: E501
        )
        ensure_file_downloaded(source_url=url, target_path=target_path)

        instances: List[Instance] = []
        with open(target_path) as f:
            for line in f:
                row = json.loads(line)
                instance_id = row["financebench_id"]
                question = row["question"]
                answer = row["answer"]
                evidence = row["evidence"][0]["evidence_text_full_page"]
                input_text = f"Evidence: {evidence}\nQuestion: {question}"
                input = Input(text=input_text)
                references = [Reference(output=Output(text=answer), tags=[CORRECT_TAG])]
                instance = Instance(id=instance_id, input=input, references=references, split=TEST_SPLIT)
                instances.append(instance)
        random.seed(0)
        train_indexes = random.sample(list(range(len(instances))), k=10)
        for train_index in train_indexes:
            instances[train_index] = dataclasses.replace(instances[train_index], split=TRAIN_SPLIT)
        return instances
