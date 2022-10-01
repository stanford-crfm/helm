import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Set

from common.hierarchical_logger import hlog, htrack, htrack_block

"""
Generates data for GPT-Fewshot.

Usage:

  python3 scripts/generate_few_shot_data.py benchmark_output/runs/fewshot

"""


@htrack("Generating data for GPT-FewShot")
def generate(suite_path: str, output_path: str):
    def write_prompts(path: str, prompts: List[str]):
        with open(path, "w") as out_file:
            for prompt in prompts:
                out_file.write(f"{{'text': {repr(prompt)}}}\n")

    # Ensures unique prompts are generated
    unique_prompts: Set[str] = set()

    for run_dir in os.listdir(suite_path):
        run_path: str = os.path.join(suite_path, run_dir)

        if not os.path.isdir(run_path):
            continue

        scenario_state_path: str = os.path.join(run_path, "scenario_state.json")
        if not os.path.isfile(scenario_state_path):
            hlog(f"WARNING: missing scenario_state.json for {run_dir}")
            continue

        with htrack_block(f"Extracting prompts from {run_dir}"):
            with open(scenario_state_path) as f:
                scenario_state: Dict = json.load(f)
                request_states: List[Dict] = scenario_state["request_states"]

                for request_state in request_states:
                    request: Dict[str, Any] = request_state["request"]
                    unique_prompts.add(request["prompt"])
                hlog(f"Processed {len(request_states)} requests.")

    with htrack_block(f"Generating dataset files to {output_path} with {len(unique_prompts)} prompts"):
        all_prompts: List[str] = list(unique_prompts)
        random.seed(0)
        random.shuffle(all_prompts)

        train_size = math.ceil(len(all_prompts) * 0.98)
        write_prompts(os.path.join(output_path, "gpt_fewshot_train.jsonl"), all_prompts[:train_size])
        write_prompts(os.path.join(output_path, "gpt_fewshot_val.jsonl"), all_prompts[train_size:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_path", type=str, help="Path to suite")
    parser.add_argument("--output-path", type=str, default=".", help="Output path")
    args = parser.parse_args()

    generate(args.suite_path, os.path.expanduser(args.output_path))
    hlog("Done.")
