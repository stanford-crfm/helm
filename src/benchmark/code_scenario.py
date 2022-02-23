"""Code scenario.

Includes
    - HumanEval: https://github.com/openai/human-eval
    - APPS: https://github.com/hendrycks/apps
"""
import gzip
import io
import json
import os
from typing import List, Dict, Iterable

from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .code_scenario_helper import run as run_reindent
from .scenario import (
    Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG
)


# === HumanEval ===
def _read_and_preprocess_human_eval(
    target_path: str,
    num_train_instances: int, num_val_instances: int, num_test_instances: int
) -> List[Instance]:
    problems = _read_human_eval(target_path)
    instances = []
    for sample_idx, task_id in enumerate(problems):
        if sample_idx < num_train_instances:
            tag = TRAIN_TAG
        elif sample_idx < num_train_instances + num_val_instances:
            tag = VALID_TAG
        else:
            tag = TEST_TAG

        instance = Instance(
            input=problems[task_id]["prompt"],
            references=[
                Reference(
                    output=problems[task_id]["canonical_solution"],
                    data=problems[task_id],
                    tags=[CORRECT_TAG],
                ),
            ],
            tags=[tag],
        )
        instances.append(instance)
    return instances


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


# === APPS ===
def _read_and_preprocess_apps(target_path: str) -> List[Instance]:
    """Read APPS dataset.

    Adapted from https://github.com/lxuechen/apps/blob/main/train/dataset_apps/APPSBaseDataset.py
    """
    instances = []
    for split, tag in zip(('train', 'test'), (TRAIN_TAG, TEST_TAG)):
        split_dir = os.path.join(target_path, split)

        skipped_problems = []
        for problem_name in os.listdir(split_dir):
            problem_dir = os.path.join(split_dir, problem_name)

            question_fname = os.path.join(problem_dir, "question.txt")
            sols_fname = os.path.join(problem_dir, "solutions.json")
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue

            # Answer type.
            starter_code_fname = os.path.join(problem_dir, "starter_code.py")
            if os.path.exists(starter_code_fname):
                answer_type = "\nUse Call-Based format\n"
            else:
                answer_type = "\nUse Standard Input format\n"

            # Starter code.
            if os.path.isfile(starter_code_fname):
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            # Question description.
            with open(question_fname, 'r') as f:
                question = f.read()

            # Solutions. Multiple of them!
            with open(sols_fname, 'r') as f:
                sols_str_list = json.load(f)
                solutions = [_reindent_code(sol_str) for sol_str in sols_str_list]

            # TODO: Truncate long instance.
            # TODO: Add input-output pairs.
            prompt = _make_input_for_apps(
                question=question, starter_code=starter_code, answer_type=answer_type
            )
            instance = Instance(
                input=prompt,
                references=[
                    Reference(output=solution, tags=[CORRECT_TAG])
                    for solution in solutions
                ],
                tags=[tag],
            )
            instances.append(instance)
        hlog(
            f"Split {split}, skipped {len(skipped_problems)} problems with no description or solution."
            f"\nTheir ids are: {skipped_problems}"
        )
    return instances


def _reindent_code(codestr):
    """Given code string, reindent it in the same way that the Github dataset was indented"""
    codestr = io.StringIO(codestr)
    ret = io.StringIO()
    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )
    return ret.getvalue()


def _make_input_for_apps(question: str, starter_code: str, answer_type: str) -> str:
    """Format the prompt as in the original training pipeline."""
    return (
        "\nQUESTION:\n" + question + "\n" + starter_code + "\n" + answer_type + "\nANSWER:\n"
    )


class CodeScenario(Scenario):
    name = "code"
    description = "Code Generation"
    tags = ["Reasoning", "Code Generation"]

    def __init__(self, dataset: str):
        self.dataset = dataset

        self.human_eval_hparams = dict(
            num_train_instances=4,
            num_val_instances=60,
            num_test_instances=100
        )

    def get_instances(self) -> List[Instance]:
        # By construction, self.output_path == 'benchmark_output/scenarios/code'.
        if self.dataset == "HumanEval":
            target_path = os.path.join(self.output_path, "HumanEval.jsonl.gz")
            ensure_file_downloaded(
                source_url="https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
                target_path=target_path,
                unpack=False,
            )
            instances = _read_and_preprocess_human_eval(target_path=target_path, **self.human_eval_hparams)

        elif self.dataset == "APPS":
            # `target_path` is the output folder, not the compressed file, since we unpack!
            target_path = os.path.join(self.output_path, "APPS")
            ensure_file_downloaded(
                source_url="https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz",
                target_path=target_path,
                unpack=True,
            )
            # TODO: Control the validation split.
            instances = _read_and_preprocess_apps(target_path)

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        return instances
