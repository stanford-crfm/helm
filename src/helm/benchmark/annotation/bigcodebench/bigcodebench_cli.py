import argparse
import json
import logging
import multiprocessing
import os
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Tuple
import gc

import numpy as np
from bigcodebench.data import get_bigcodebench, get_bigcodebench_hash, load_solutions
from bigcodebench.data.utils import CACHE_DIR
from bigcodebench.eval import PASS, estimate_pass_at_k, untrusted_check
from bigcodebench.gen.util import trusted_check

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Result = Tuple[str, List[bool]]


def get_groundtruth(
    n_workers, problems, hashcode, check_gt_only, max_as_limit, max_data_limit, max_stack_limit, min_time_limit
):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    tbegin = time.time()  # noqa: F841

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        n_samples = 0
        expected_time = dict()

        for problem in problems.values():
            args = (
                problem["complete_prompt"] + "\n" + problem["canonical_solution"],
                problem["test"],
                problem["task_id"],
                max_as_limit,
                max_data_limit,
                max_stack_limit,
                min_time_limit,
            )

            futures.append(executor.submit(trusted_check, *args))
            n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            expected_time[result["task_id"]] = result["time"]

    if any(expected_time.values()):
        with open(cache_file, "wb") as f:
            pickle.dump(expected_time, f)

    return expected_time


def check_correctness(
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    identifier=None,
    min_time_limit: float = 0.1,
    gt_time_limit: float = 2.0,
) -> Dict[str, Result]:
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        solution,
        problem["test"],
        problem["entry_point"],
        max_as_limit,
        max_data_limit,
        max_stack_limit,
        min_time_limit,
        gt_time_limit,
    )
    return ret


def evaluate(
    split: str,
    subset: str,
    samples: str,
    pass_k: str = "1,5,10",
    parallel: int = -1,
    min_time_limit: float = 1,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    calibrated: bool = True,
    check_gt_only: bool = False,
    no_gt: bool = False,
    selective_evaluate: str = "",
    output_file: str = "output.json",
):
    passk = [int(k.strip()) for k in pass_k.split(",") if k.strip().isdigit()]
    if parallel < 1:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = parallel

    if check_gt_only:
        samples = "__dummy__.jsonl"

    extra = subset + "_" if subset != "full" else ""  # noqa: F841

    problems = get_bigcodebench(subset=subset)

    # Add selective evaluation logic
    if selective_evaluate:
        selected_ids = ["BigCodeBench/" + id for id in sorted(set(selective_evaluate.split(",")))]
        problems = {k: v for k, v in problems.items() if k in selected_ids}
        if not problems:
            raise ValueError(f"None of the provided task IDs {selected_ids} were found in the dataset")

    dataset_hash = get_bigcodebench_hash(subset=subset)

    if not no_gt:
        expected_time = get_groundtruth(
            n_workers,
            problems,
            dataset_hash,
            check_gt_only,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            min_time_limit,
        )
    else:
        expected_time = {task_id: None for task_id in problems}

    gt_pass_rate = np.mean([1 if v is not None else 0 for k, v in expected_time.items() if k in problems])
    failed_tasks = [k for k, v in expected_time.items() if v is None and k in problems]

    pass_at_k = dict()
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": {},
    }

    if not check_gt_only:
        logger.info(f"Evaluating samples from {samples}")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()  # type: ignore
            n_samples = 0
            eval_results = defaultdict(list)
            remainings = set()

            for sample in load_solutions(samples):
                task_id = sample["task_id"]

                if task_id not in problems:
                    continue
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["complete_prompt"] + sample["completion"]
                )
                if calibrated:
                    solution = problems[task_id]["code_prompt"] + "\n    pass\n" + solution
                remainings.add(sample["_identifier"])
                args = (
                    completion_id[task_id],
                    problems[task_id],
                    solution,
                    max_as_limit,
                    max_data_limit,
                    max_stack_limit,
                    sample["_identifier"],
                    min_time_limit,
                    expected_time[task_id] if expected_time[task_id] else 20,
                )
                futures.append(executor.submit(check_correctness, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"
            assert len(completion_id) == len(problems), "Missing problems in samples"

            for future in as_completed(futures):
                result = future.result()
                remainings.remove(result["_identifier"])
                eval_results[result["task_id"]].append(result)
                del future, result
                gc.collect()

        # sort the results for each problem by completion_id
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []  # type: ignore
            for res in task_results:
                stat, details = res["base"]
                results["eval"][task_id].append(  # type: ignore
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "status": stat,
                        "details": details,
                    }
                )

        # Calculate pass@k.
        total = np.array([len(r) for k, r in results["eval"].items() if k in problems])  # type: ignore
        base_correct = []

        for key, res in results["eval"].items():  # type: ignore
            if key not in problems:
                continue
            bc = sum([r["status"] == PASS for r in res])  # type: ignore
            base_correct.append(bc)

        base_correct = np.array(base_correct)  # type: ignore

        pass_at_k.update(
            {f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean() for k in passk if total.min() >= k}
        )

        del problems, futures
        gc.collect()

    pass_at_k["model"] = os.path.basename(samples).split("--bigcodebench-")[0]
    pass_at_k["split"] = split
    pass_at_k["subset"] = subset
    pass_at_k["calibrated"] = calibrated
    pass_at_k["gt_pass_rate"] = gt_pass_rate
    pass_at_k["failed_tasks"] = failed_tasks

    # Save results to output file
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump({"results": results, "metrics": pass_at_k}, f, indent=4)
        logger.info(f"Results saved to {output_file}")

    return results, pass_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["complete", "instruct"], default="complete")
    parser.add_argument("--subset", type=str, choices=["full", "hard"], default="full")
    parser.add_argument("--samples", type=str, required=False, default="samples.jsonl")
    parser.add_argument("--pass-k", type=str, default="1,5,10")
    parser.add_argument("--parallel", type=int, default=-1)
    parser.add_argument("--min-time-limit", type=float, default=1)
    parser.add_argument("--max-as-limit", type=int, default=30 * 1024)
    parser.add_argument("--max-data-limit", type=int, default=30 * 1024)
    parser.add_argument("--max-stack-limit", type=int, default=10)
    parser.add_argument("--calibrated", action="store_true", default=True)
    parser.add_argument("--no-calibrated", action="store_false", dest="calibrated")
    parser.add_argument("--check-gt-only", action="store_true", default=False)
    parser.add_argument("--no-gt", action="store_true", default=False)
    parser.add_argument("--selective-evaluate", type=str, default="")
    parser.add_argument("--output", type=str, default="output.json")

    args = parser.parse_args()

    results, metrics = evaluate(
        split=args.split,
        subset=args.subset,
        samples=args.samples,
        pass_k=args.pass_k,
        parallel=args.parallel,
        min_time_limit=args.min_time_limit,
        max_as_limit=args.max_as_limit,
        max_data_limit=args.max_data_limit,
        max_stack_limit=args.max_stack_limit,
        calibrated=args.calibrated,
        check_gt_only=args.check_gt_only,
        no_gt=args.no_gt,
        selective_evaluate=args.selective_evaluate,
        output_file=args.output,
    )

    # Print metrics summary to console for potential docker debugging
    print("\n===== Evaluation Results =====")
    for k, v in metrics.items():
        if k == "failed_tasks" and v:
            print(f"{k}: {v}")
        elif k == "failed_tasks":
            print(f"{k}: None")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()