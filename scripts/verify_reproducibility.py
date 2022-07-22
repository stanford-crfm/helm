import argparse
import json
import os
import subprocess
from typing import List, Optional

from jsoncomparison import Compare, NO_DIFF

from common.hierarchical_logger import hlog, htrack, htrack_block

"""
Verifies that the Scenario construction and generation of prompts are reproducible:

1. Performs dryrun
2. Performs dryrun again.
3. Compares the requests in the scenario_state.json files of the two dryrun output folders.

Usage:

  python3 scripts/verify_reproducibility.py --models-to-run openai/davinci openai/code-cushman-001 together/gpt-neox-20b

"""
DRYRUN_SUITE1: str = "dryrun1"
DRYRUN_SUITE2: str = "dryrun2"


@htrack("Performing dryrun")
def do_dry_run(
    dryrun_suite: str, conf_path: str, max_eval_instances: int, priority: int, models: Optional[List[str]]
) -> str:
    """Performs dry run. Blocks until the run finishes."""
    command: List[str] = [
        "benchmark-present",
        f"--suite={dryrun_suite}",
        f"--conf-path={conf_path}",
        f"--max-eval-instances={max_eval_instances}",
        "--local",
        "--dry-run",
        f"--priority={priority}",
    ]
    if models:
        command.append("--models-to-run")
        command.extend(models)

    hlog(" ".join(command))
    subprocess.call(command)
    output_path: str = f"benchmark_output/runs/{dryrun_suite}"
    hlog(f"Results written out to path: {output_path}.")
    return output_path


@htrack("Verifying reproducibility")
def verify_reproducibility(conf_path: str, max_eval_instances: int, priority: int, models: Optional[List[str]]):
    dryrun_path1: str = do_dry_run(DRYRUN_SUITE1, conf_path, max_eval_instances, priority, models)
    dryrun_path2: str = do_dry_run(DRYRUN_SUITE2, conf_path, max_eval_instances, priority, models)

    hlog(f"Comparing results in {dryrun_path1} vs. {dryrun_path2}")
    for run_dir in os.listdir(dryrun_path1):
        run_path1: str = os.path.join(dryrun_path1, run_dir)

        if not os.path.isdir(run_path1):
            continue

        scenario_state_path1: str = os.path.join(run_path1, "scenario_state.json")
        if not os.path.isfile(scenario_state_path1):
            continue

        run_path2: str = os.path.join(dryrun_path2, run_dir)
        scenario_state_path2: str = os.path.join(run_path2, "scenario_state.json")

        with htrack_block(f"Comparing `ScenarioState`s for {run_dir}"):
            with open(scenario_state_path1) as f:
                scenario_state1 = json.load(f)

            with open(scenario_state_path2) as f:
                scenario_state2 = json.load(f)

            diff = Compare().check(scenario_state1, scenario_state2)
            if diff != NO_DIFF:
                hlog(f"ERROR: not reproducible. Diff:\n{diff}")
            else:
                hlog("Verified reproducibility.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--max-eval-instances", type=int, default=1000, help="Maximum number of eval instances.",
    )
    parser.add_argument(
        "-c",
        "--conf-path",
        type=str,
        help="Where to read RunSpecs to run from",
        default="src/benchmark/presentation/run_specs.conf",
    )
    parser.add_argument(
        "--models-to-run",
        nargs="+",
        help="Only RunSpecs with these models specified. If no model is specified, run with all models.",
        default=None,
    )
    parser.add_argument(
        "--priority", type=int, default=2, help="Run RunSpecs with priority less than or equal to this number."
    )
    args = parser.parse_args()

    verify_reproducibility(args.conf_path, args.max_eval_instances, args.priority, args.models_to_run)
    hlog("Done.")
