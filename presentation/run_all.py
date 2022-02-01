import argparse
import os.path
import subprocess
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Set

from pyhocon import ConfigFactory, ConfigTree

"""
Script that runs all the RunSpecs in run_specs.conf and outputs a status page.

Usage:
    
    python3 presentation/run_all.py -o <Where to output benchmarking results> -s <Where to output status page>

"""

# For READY RunSpecs, evaluate and generate metrics.
READY_STATE = "READY"

# For WIP RunSpecs, just estimate token usage.
WIP_STATE = "WIP"


class Runner:
    BENCHMARK_RUN_EXECUTABLE: str = "venv/bin/benchmark-run"

    @staticmethod
    def get_run_spec_description(run_spec_name: str, run_spec_args: Optional[ConfigTree]) -> str:
        """Get full the run spec description by appending args to the RunSpec name."""
        if run_spec_args:
            args_str: str = ",".join([f"{arg}={run_spec_args[arg]}" for arg in run_spec_args])
            return f"{run_spec_name}:{args_str}"
        else:
            return run_spec_name

    def __init__(self, url: str, api_key_path: str, output_path: str, status_path: str, num_threads: int):
        self.url: str = url
        self.api_key_path: str = api_key_path
        self.output_path: str = output_path
        self.status_path: str = status_path
        self.num_threads: int = num_threads

    def run(self):
        print("Reading RunSpecs from run_specs.conf...")
        conf = ConfigFactory.parse_file("presentation/run_specs.conf")
        ready_run_specs: Set[str] = set()

        print("Running all RunSpecs...")
        for run_spec_name, run_spec in tqdm(conf.items()):
            state: str = run_spec.state
            run_spec_description: str = Runner.get_run_spec_description(
                run_spec_name, run_spec_args=run_spec.get("args", default=None),
            )

            if state == READY_STATE:
                # Folders and filenames with ":" will be replaced with "_"
                ready_run_specs.add(run_spec_description.replace(":", "_"))
                self.run_benchmarking(run_spec_description)
            elif state == WIP_STATE:
                self.run_benchmarking(run_spec_description, dry_run=True)
            else:
                raise ValueError(f"RunSpec {run_spec_name} has an invalid state: {state}")

        # Create the status page by traversing through and extracting metrics from the benchmarking output files
        print("Creating the status page...")
        wip_content: List[str] = [
            "##### Work In Progress Run Specs #####",
            "",
            "The following RunSpecs are not ready for evaluation. Instead of evaluating, "
            "we just estimate the number of tokens needed.",
            "",
        ]
        ready_content: List[str] = ["##### Ready Run Specs ######"]

        scenarios_dir: str = os.path.join(self.output_path, "scenarios")
        for scenario in os.listdir(scenarios_dir):
            scenario_dir: str = os.path.join(scenarios_dir, scenario)
            if not os.path.isdir(scenario_dir):
                continue

            runs_dir: str = os.path.join(scenario_dir, "runs")
            for run in os.listdir(runs_dir):
                run_dir: str = os.path.join(runs_dir, run)
                if not os.path.isdir(run_dir):
                    continue

                content: List[str] = ready_content if run in ready_run_specs else wip_content
                metrics_text: str = Path(os.path.join(run_dir, "metrics.txt")).read_text()
                content.append(f"{run.replace('_', ':', 1)} - {metrics_text}")

        # Write out the status page
        with open(self.status_path, "w") as f:
            # Write out the WIP RunSpecs first
            f.write("\n".join(wip_content))
            f.write("\n" * 2 + "-" * 150 + "\n" * 2)
            f.write("\n".join(ready_content))

    def run_benchmarking(self, run_spec_description: str, dry_run: bool = False):
        command: List[str] = [
            Runner.BENCHMARK_RUN_EXECUTABLE,
            f"--url {self.url}",
            f"--api-key-path {self.api_key_path}",
            f"--output-path {self.output_path}",
            f"--num-threads {self.num_threads}",
            f"--run-specs {run_spec_description}",
        ]
        if dry_run:
            command.append("--dry-run")

        try:
            subprocess.check_call(
                " ".join(command), shell=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"There was an error while creating the temp instance: {e}")
            raise


def main():
    runner = Runner(
        url=args.url,
        api_key_path=args.api_key_path,
        output_path=args.output_path,
        status_path=args.status_path,
        num_threads=args.num_threads,
    )
    runner.run()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="https://crfm-models.stanford.edu",
        help="URL of the instance to use when benchmarking",
    )
    parser.add_argument(
        "-a", "--api-key-path", type=str, default="proxy_api_key.txt", help="Path to API key",
    )
    parser.add_argument(
        "-o", "--output-path", help="Where to save all the benchmarking output", default="benchmark_output",
    )
    parser.add_argument(
        "-s",
        "--status-path",
        help="Where to output current status of the Benchmarking project",
        default="benchmark_output/status.txt",
    )
    parser.add_argument(
        "-n", "--num-threads", type=int, help="Max number of threads to make requests", default=5,
    )
    args = parser.parse_args()

    main()
