import csv
import os
import gzip
import json
from typing import List, Dict, Iterable
from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG


def read_problems(evalset_file: str = "HumanEval.jsonl.gz") -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


class CodeScenario(Scenario):
    """
    The Code Generation task HumanEval:

        https://arxiv.org/pdf/2107.03374.pdf

    Code is adapted from:

        https://github.com/openai/human-eval
    """

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

            # Read all the instances
            problems = read_problems(data_path)
        else:
            raise "Not exists"
        
        instances = []            
        for sample_idx, task_id in enumerate(problems):
            if sample_idx < self.num_train_instances:
                cur_tag = TRAIN_TAG
            elif sample_idx < self.num_train_instances + self.num_val_instances:
                cur_tag = VALID_TAG
            else:
                cur_tag = TEST_TAG
            instance = Instance(
                input=problems[task_id]["prompt"], references=[
                    Reference(output=problems[task_id]["canonical_solution"], tags=[CORRECT_TAG]),
                    Reference(output=problems[task_id]["test"], tags=[CORRECT_TAG]),
                    Reference(output=problems[task_id]["entry_point"], tags=[CORRECT_TAG]),
                ], tags=[cur_tag],
            )
            instances.append(instance)

        return instances
