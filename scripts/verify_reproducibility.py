import argparse
import os
import shutil
import subprocess
from typing import List, Optional

from helm.common.hierarchical_logger import hlog, htrack, htrack_block

"""
Verifies that the Scenario construction and generation of prompts are
reproducible between runs:

1. Performs a dry-run if --source-suite isn't specified.
2. Performs a dry-run again if --target-suite isn't specified.
3. Compares the requests in the scenario_state.json files of the two folders.

Usage:

  python3 scripts/verify_reproducibility.py --models-to-run openai/davinci openai/code-cushman-001 together/gpt-neox-20b

"""
OUTPUT_PATH_TEMPLATE = "benchmark_output/runs/{suite}"
DRYRUN_SUITE1: str = "dryrun_results1"
DRYRUN_SUITE2: str = "dryrun_results2"


@htrack("Performing dryrun")
def do_dry_run(
    dryrun_suite: str, conf_path: str, max_eval_instances: int, priority: int, models: Optional[List[str]]
) -> str:
    """Performs dry run. Blocks until the run finishes."""
    output_path: str = OUTPUT_PATH_TEMPLATE.format(suite=dryrun_suite)
    shutil.rmtree(output_path, ignore_errors=True)
    hlog(f"Deleted old results at path: {output_path}.")

    command: List[str] = [
        "helm-run",
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
    hlog(f"Results are written out to path: {output_path}.")
    return output_path


@htrack("Verifying reproducibility")
def verify_reproducibility(
    source_suite: Optional[str],
    target_suite: Optional[str],
    conf_path: str,
    max_eval_instances: int,
    priority: int,
    models: Optional[List[str]],
):
    output_path1: str = (
        OUTPUT_PATH_TEMPLATE.format(suite=source_suite)
        if source_suite
        else do_dry_run(DRYRUN_SUITE1, conf_path, max_eval_instances, priority, models)
    )
    output_path2: str = (
        OUTPUT_PATH_TEMPLATE.format(suite=target_suite)
        if target_suite
        else do_dry_run(DRYRUN_SUITE2, conf_path, max_eval_instances, priority, models)
    )

    hlog(f"Comparing results in {output_path1} vs. {output_path2}")
    for run_dir in os.listdir(output_path1):
        run_path1: str = os.path.join(output_path1, run_dir)

        if not os.path.isdir(run_path1):
            continue

        scenario_state_path1: str = os.path.join(run_path1, "scenario_state.json")
        if not os.path.isfile(scenario_state_path1):
            continue

        run_path2: str = os.path.join(output_path2, run_dir)
        scenario_state_path2: str = os.path.join(run_path2, "scenario_state.json")

        with htrack_block(f"Comparing `ScenarioState`s for {run_dir}"):
            with open(scenario_state_path1) as f:
                scenario_state1 = f.readlines()

            with open(scenario_state_path2) as f:
                scenario_state2 = f.readlines()

            same: bool = True
            # Check the difference between two scenario_state.json files
            for i, (line1, line2) in enumerate(zip(scenario_state1, scenario_state2)):
                if line1 != line2:
                    line_number: int = i + 1
                    same = False
                    hlog(
                        "ERROR: Not reproducible - content of "
                        f"{scenario_state_path1} and {scenario_state_path2} are different. "
                        f"Line {line_number}:"
                    )
                    hlog(f"--- scenario_state.json (1): {line1}")
                    hlog(f"+++ scenario_state.json (2): {line2}")
                    break

            if same:
                hlog("Verified reproducible.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--max-eval-instances",
        type=int,
        default=1000,
        help="Maximum number of eval instances.",
    )
    parser.add_argument(
        "-c",
        "--conf-path",
        type=str,
        help="Where to read RunSpecs to run from",
        default="src/helm/benchmark/presentation/run_entries.conf",
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
    parser.add_argument(
        "-s",
        "--source-suite",
        type=str,
        default=None,
        help="Source suite that will be compared.",
    )
    parser.add_argument(
        "-t",
        "--target-suite",
        type=str,
        default=None,
        help="Target suite that will be compared.",
    )
    args = parser.parse_args()

    verify_reproducibility(
        args.source_suite, args.target_suite, args.conf_path, args.max_eval_instances, args.priority, args.models_to_run
    )
    hlog("Done.")
