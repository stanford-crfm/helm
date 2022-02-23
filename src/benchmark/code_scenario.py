"""Scenario related to source code.

Includes
    - HumanEval: https://github.com/openai/human-eval
    - APPS: https://github.com/hendrycks/apps
"""
import gzip
import json
import os
from typing import List, Dict, Iterable

from common.general import ensure_file_downloaded
from .scenario import (
    Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG
)


def _read_human_eval(evalset_file: str = "HumanEval.jsonl.gz") -> Dict[str, Dict]:
    return {task["task_id"]: task for task in _stream_jsonl(evalset_file)}


def _stream_jsonl(filename: str) -> Iterable[Dict]:
    """Parses each jsonl line and yields it as a dictionary."""
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def _read_apps(filename: str) -> Dict[str, Dict]:
    raise NotImplemented


class CodeScenario(Scenario):
    name = "code"
    description = "Code Generation"
    tags = ["Reasoning", "Code Generation"]

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.num_train_instances: int = 4
        self.num_val_instances: int = 60
        self.num_test_instances: int = 100

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        data_path = os.path.join(self.output_path, "HumanEval.jsonl.gz")
        if self.dataset == "HumanEval":
            ensure_file_downloaded(
                source_url="https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
                target_path=data_path,
                unpack=False,
            )

            problems = _read_human_eval(data_path)

            instances = []
            for sample_idx, task_id in enumerate(problems):
                if sample_idx < self.num_train_instances:
                    cur_tag = TRAIN_TAG
                elif sample_idx < self.num_train_instances + self.num_val_instances:
                    cur_tag = VALID_TAG
                else:
                    cur_tag = TEST_TAG
                instance = Instance(
                    input=problems[task_id]["prompt"],
                    references=[
                        Reference(
                            output=problems[task_id]["canonical_solution"], data=problems[task_id], tags=[CORRECT_TAG]
                        ),
                    ],
                    tags=[cur_tag],
                )
                instances.append(instance)

        elif self.dataset == "apps":
            ensure_file_downloaded(
                source_url="https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz",
                target_path=data_path,
                unpack=False,
            )
            # TODO:
            instances = []
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        return instances
