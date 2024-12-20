"""Code scenario.

Includes
    - HumanEval: https://github.com/openai/human-eval
    - APPS: https://github.com/hendrycks/apps

HumanEval is a small dataset of human written test cases. Each instance has
1) a prompt, 2) a canonical_solution, and 3) test cases. Here's one example
taken from the dataset:

1) prompt:

    from typing import List


    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        '''Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        '''

2) canonical_solution:

    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False

3) test cases:

    def check(candidate):
        assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
        assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
        assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
        assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
        assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
        assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
        assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

APPS is a benchmark for code generation from natural language specifications.
Each instance has 1) a problem description with examples (as what you get in
programming competitions), 2) coding solutions, 3) test cases.
"""

import io
import json
import os
import sys
from typing import List, Dict, Iterable, Optional, cast

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.code_scenario_helper import run as run_reindent
from helm.benchmark.scenarios.code_scenario_apps_pinned_file_order import apps_listdir_with_pinned_order
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class CodeReference(Reference):
    # Extra none-string metadata, e.g., paths.
    test_cases: Optional[Dict] = None

    def __init__(self, test_cases=None, **kw):
        self.test_cases = test_cases
        super(CodeReference, self).__init__(**kw)


class CodeInstance(Instance):
    reference: CodeReference

    # Extra none-string metadata, e.g., paths.
    metadata: Optional[Dict] = None

    def __init__(self, metadata=None, **kw):
        self.metadata = metadata
        super(CodeInstance, self).__init__(**kw)


# === HumanEval ===
def _read_and_preprocess_human_eval(
    target_path: str, num_train_instances: int, num_val_instances: int, num_test_instances: int
) -> List[CodeInstance]:
    problems = _read_human_eval(target_path)
    instances = []
    for sample_idx, task_id in enumerate(problems):
        if sample_idx < num_train_instances:
            split = TRAIN_SPLIT
        elif sample_idx < num_train_instances + num_val_instances:
            split = VALID_SPLIT
        else:
            split = TEST_SPLIT

        instance = CodeInstance(
            input=Input(text=problems[task_id]["prompt"]),
            references=[
                CodeReference(
                    output=Output(text=problems[task_id]["canonical_solution"]),
                    test_cases=problems[task_id],
                    tags=[CORRECT_TAG],
                )
            ],
            split=split,
        )
        instances.append(instance)
    return instances


def _read_human_eval(evalset_file: str = "HumanEval.jsonl") -> Dict[str, Dict]:
    return {task["task_id"]: task for task in _stream_jsonl(evalset_file)}


def _stream_jsonl(filename: str) -> Iterable[Dict]:
    """Parses each jsonl line and yields it as a dictionary."""
    with open(filename, "r") as f:
        for line in f:
            if any(not x.isspace() for x in line):
                yield json.loads(line)


# === APPS ===
def _read_and_preprocess_apps(target_path: str) -> List[CodeInstance]:
    """Read APPS dataset.

    Adapted from
        https://github.com/lxuechen/apps/blob/main/train/dataset_apps/APPSBaseDataset.py
    """
    # Some versions of Python have a configurable maximum number of digits of integers that can be parsed
    # from strings. This limit also applies to parsing integers in JSON. The default limit is 4300 digits.
    #
    # Reading APPS instances will fail with the default limit because the APPS dataset contains very large
    # integers.
    #
    # The sys.set_int_max_str_digits() method can be used to increase the limit. This method exists if and
    # only if the version of Python has a default limit.
    #
    # See: https://docs.python.org/3/library/stdtypes.html#int-max-str-digits
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(100000)

    SINGLE_STR_LIMIT = 150000  # From original codebase.

    instances = []
    for split_tag in (TRAIN_SPLIT, TEST_SPLIT):
        split_dir = os.path.join(target_path, split_tag)

        num_problems = 0
        skipped_problems = []
        for problem_name in apps_listdir_with_pinned_order(target_path, split_tag):
            problem_dir = os.path.join(split_dir, problem_name)

            question_fname = os.path.join(problem_dir, "question.txt")
            sols_fname = os.path.join(problem_dir, "solutions.json")
            tests_fname = os.path.join(problem_dir, "input_output.json")

            # All instances must have the question description.
            if not os.path.isfile(question_fname):
                skipped_problems.append(problem_name)
                continue
            else:
                # Train instances must have solution code.
                if split_tag in ("train",):
                    if not os.path.isfile(sols_fname):
                        skipped_problems.append(problem_name)
                        continue
                # Test instances can ignore solution code, but must have test cases.
                elif split_tag in ("test",):
                    if not os.path.exists(tests_fname) or not os.path.isfile(tests_fname):
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
                with open(starter_code_fname, "r") as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            # Question description.
            with open(question_fname, "r") as f:
                question = f.read()

            # Solutions. Multiple of them!
            if os.path.isfile(sols_fname):
                with open(sols_fname, "r") as f:
                    sols_str_list = json.load(f)
                    solutions = [_reindent_code(sol_str) for sol_str in sols_str_list]
            else:
                solutions = []

            # Tests.
            if os.path.exists(tests_fname):
                with open(tests_fname, "r") as f:
                    # Some files may contain the key `fn_name`, which indicates it's
                    # call-based instance. Annoying!

                    # Call-based instances check function input/outputs; for other instances
                    # I/O is handled through stdin and stdout streams.
                    data: Dict = json.load(f)
            else:
                data = dict()
            data["root"] = problem_dir

            # Truncate for training, following original codebase.
            question = question[:SINGLE_STR_LIMIT]
            starter_code = starter_code[:SINGLE_STR_LIMIT]
            solutions = [sol[:SINGLE_STR_LIMIT] for sol in solutions]
            if len(solutions) == 0:
                solutions = [""]

            # Create overall prompt.
            prompt = _make_input_for_apps(
                question=question,
                starter_code=starter_code,
                answer_type=answer_type,
            )
            instance = CodeInstance(
                input=Input(text=prompt),
                references=[
                    CodeReference(output=Output(text=solution), tags=[CORRECT_TAG], test_cases=data)
                    for solution in solutions
                ],
                split=split_tag,
                metadata=data,
            )
            instances.append(instance)
            num_problems += 1
        # Should not skip any cases; just defensive.
        hlog(
            f"Split {split_tag}, "
            f"skipped {len(skipped_problems)}/{num_problems} problems with no description or solution. "
            f"Their ids are: {skipped_problems}"
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
            "all-tabs": False,
        },
    )
    return ret.getvalue()


def _make_input_for_apps(question: str, starter_code: str, answer_type: str) -> str:
    """Format the prompt as in the original training pipeline."""
    # Different from the original paper: We add the phrase 'in Python' to make models only generate Python code;
    #   otherwise models can generate C++ and code in other languages. The evaluation engine, mostly copied from the
    #   original APPS codebase, runs PyExt and has no way to execute C++ code.
    # The extra phrase isn't needed when there's in-context examples of Python code.
    return "\nQUESTION:\n" + question + "\n" + starter_code + "\n" + answer_type + "\nANSWER in Python code:\n"

    # Below is what's used in the original paper for reference and comparison.
    # return "\nQUESTION:\n" + question + "\n" + starter_code + "\n" + answer_type + "\nANSWER:\n"


class CodeScenario(Scenario):
    name = "code"
    description = "Code Generation"
    tags = ["Reasoning", "Code Generation"]

    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset

        self.human_eval_hparams = dict(num_train_instances=0, num_val_instances=0, num_test_instances=164)

    def get_instances(self, output_path: str) -> List[Instance]:
        # By construction, output_path == args.output_path + '/scenarios/code'
        # where args.output_path is parsed from the command line argument.
        # The default self.output_path here is '/benchmark_output/scenarios/ice'.
        # See helm.benchmark.runner for more details about args.output_path.
        if self.dataset == "humaneval":
            target_path = os.path.join(output_path, "HumanEval.jsonl")
            ensure_file_downloaded(
                source_url="https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
                target_path=target_path,
                unpack=False,
            )
            instances = _read_and_preprocess_human_eval(target_path=target_path, **self.human_eval_hparams)

        elif self.dataset == "apps":
            # This dataset doesn't have a validation set.
            # Unclear how they do validation. Also not clarified in their paper.
            # `target_path` is the output folder, not the compressed file, since we unpack!
            target_path = os.path.join(output_path, "APPS")
            ensure_file_downloaded(
                source_url="https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz",
                target_path=target_path,
                unpack=True,
            )
            instances = _read_and_preprocess_apps(target_path)

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        return cast(List[Instance], instances)
