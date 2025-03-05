import gradio as gr
import json
import logging
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime
from typing import Any, Dict, List, Tuple
from warnings import warn
import gc

import numpy as np
from huggingface_hub import HfApi
from bigcodebench.data import get_bigcodebench, get_bigcodebench_hash, load_solutions
from bigcodebench.data.utils import CACHE_DIR
from bigcodebench.eval import PASS, compatible_eval_result, estimate_pass_at_k, untrusted_check
from bigcodebench.gen.util import trusted_check
from apscheduler.schedulers.background import BackgroundScheduler

REPO_ID = "bigcode/bigcodebench-evaluator"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
API = HfApi(token=HF_TOKEN)
Result = Tuple[str, List[bool]]


def get_groundtruth(n_workers, problems, hashcode, check_gt_only, max_as_limit, max_data_limit, max_stack_limit, min_time_limit):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    tbegin = time.time()
    
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
    pass_k: str="1,5,10",
    parallel: int = -1,
    min_time_limit: float = 1,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    calibrated: bool = True,
    check_gt_only: bool = False,
    no_gt: bool = False,
    selective_evaluate: str = "",
):
    passk = [int(k.strip()) for k in pass_k.split(',') if k.strip().isdigit()]
    if parallel < 1:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = parallel

    if check_gt_only:
        samples = "__dummy__.jsonl"

    extra = subset + "_" if subset != "full" else ""

    problems = get_bigcodebench(subset=subset)
    
    # Add selective evaluation logic
    if selective_evaluate:
        selected_ids = ["BigCodeBench/" + id for id in sorted(set(selective_evaluate.split(",")))]
        problems = {k: v for k, v in problems.items() if k in selected_ids}
        if not problems:
            raise ValueError(f"None of the provided task IDs {selected_ids} were found in the dataset")

    dataset_hash = get_bigcodebench_hash(subset=subset)
    
    if not no_gt:
        expected_time = get_groundtruth(n_workers, problems, dataset_hash, check_gt_only, max_as_limit, max_data_limit, max_stack_limit, min_time_limit)
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

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
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
                    expected_time[task_id] if expected_time[task_id] else 20
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
            results["eval"][task_id] = []
            for res in task_results:
                stat, details = res["base"]
                results["eval"][task_id].append(
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "status": stat,
                        "details": details,
                    }
                )

        # Calculate pass@k.
        total = np.array([len(r) for k, r in results["eval"].items() if k in problems])
        base_correct = []

        for key, res in results["eval"].items():
            if key not in problems:
                continue
            bc = sum([r["status"] == PASS for r in res])
            base_correct.append(bc)

        base_correct = np.array(base_correct)

        pass_at_k.update({
            f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
            for k in passk
            if total.min() >= k
        })

        del problems, futures
        gc.collect()
        
    pass_at_k["model"] = os.path.basename(samples).split("--bigcodebench-")[0]
    pass_at_k["split"] = split
    pass_at_k["subset"] = subset
    pass_at_k["calibrated"] = calibrated
    pass_at_k["gt_pass_rate"] = gt_pass_rate
    pass_at_k["failed_tasks"] = failed_tasks
    
    return results, pass_at_k


# def run_gradio():
interface = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.Dropdown(["complete", "instruct"], label="BigCodeBench Split"),
        gr.Dropdown(["full", "hard"], label="BigCodeBench Subset"),
        gr.File(label="Samples Path (.jsonl)"),
        gr.Textbox(label="Pass k Values (comma-separated)", value="1,5,10"),
        gr.Slider(-1, multiprocessing.cpu_count(), step=1, label="Parallel Workers", value=-1),
        gr.Slider(0.1, 10, step=0.1, label="Min Time Limit", value=1),
        gr.Slider(1, 100 * 1024, step=1024, label="Max AS Limit", value=30 * 1024),
        gr.Slider(1, 100 * 1024, step=1024, label="Max Data Limit", value=30 * 1024),
        gr.Slider(1, 100, step=1, label="Max Stack Limit", value=10),
        gr.Checkbox(label="Calibrated", value=True),
        gr.Checkbox(label="Check GT Only"),
        gr.Checkbox(label="No GT"),
        gr.Textbox(label="Selective Evaluated Task IDs (comma-separated, e.g. '0,1,2')", value=""),
    ],
    outputs=[
        gr.JSON(label="Results"),
        gr.JSON(label="Eval Results"),
    ],
    # concurrency_limit=None
)
interface.queue(default_concurrency_limit=None)


def preload_gt():
    evaluate(split="complete", subset="full", samples="", check_gt_only=True)
    evaluate(split="complete", subset="hard", samples="", check_gt_only=True)


def restart_space():
    logging.info(f"Restarting space with repo ID: {REPO_ID}")
    try:
        # Now restart the space
        API.restart_space(repo_id=REPO_ID, token=HF_TOKEN)
        logging.info("Space restarted successfully.")
    except Exception as e:
        logging.error(f"Failed to restart space: {e}")


# if __name__ == "__main__":
preload_gt()
scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", hours=2)  # Restart every 2hs
scheduler.start()
interface.launch(show_error=True)
