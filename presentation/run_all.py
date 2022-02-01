import argparse
import os.path
import subprocess
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from pyhocon import ConfigFactory

"""
Script that runs all the RunSpecs in run_specs.conf and outputs a status page.

Usage:
    
    python3 presentation/run_all.py -s <Where to output the status page>

"""

# For READY RunSpecs, evaluate and generate metrics.
READY_STATUS = "READY"

# For WIP RunSpecs, just estimate token usage.
WIP_STATUS = "WIP"


class Runner:
    BENCHMARK_RUN_EXECUTABLE: str = "venv/bin/benchmark-run"

    def __init__(self, url: str, api_key_path: str, output_path: str, status_path: str, num_threads: int):
        self.url: str = url
        self.api_key_path: str = api_key_path
        self.output_path: str = output_path
        self.status_path: str = status_path
        self.num_threads: int = num_threads

    def run(self):
        print("Reading RunSpecs from run_specs.conf...")
        conf = ConfigFactory.parse_file("presentation/run_specs.conf")
        ready_run_spec_dir_to_description: Dict[str, str] = {}
        wip_run_spec_dir_to_description: Dict[str, str] = {}

        print("Running all RunSpecs...")
        for run_spec, run_spec_state in tqdm(conf.items()):
            run_spec = run_spec.replace('"', "")
            status: str = run_spec_state.status
            # Folders and filenames with ":" will be replaced with "_"
            run_spec_dir: str = run_spec.replace(":", "_")

            if status == READY_STATUS:
                ready_run_spec_dir_to_description[run_spec_dir] = run_spec
                self.run_benchmarking(run_spec)
            elif status == WIP_STATUS:
                wip_run_spec_dir_to_description[run_spec_dir] = run_spec
                self.run_benchmarking(run_spec, dry_run=True)
            else:
                raise ValueError(f"RunSpec {run_spec} has an invalid status: {status}")

        # Create the status page by traversing through and extracting metrics from the benchmarking output files
        print("Creating the status page...")
        wip_content: List[str] = [
            "##### Work In Progress Run Specs #####",
            "",
            "The following RunSpecs are not ready for evaluation. Instead of evaluating, "
            "we just estimate the number of tokens needed.",
            "",
        ]
        ready_content: List[str] = ["##### Ready Run Specs ######\n"]

        scenarios_dir: str = os.path.join(self.output_path, "scenarios")
        for scenario in os.listdir(scenarios_dir):
            scenario_dir: str = os.path.join(scenarios_dir, scenario)
            if not os.path.isdir(scenario_dir):
                continue

            run_spec_dirs: str = os.path.join(scenario_dir, "runs")
            for run_spec_dir in os.listdir(run_spec_dirs):
                full_run_spec_path: str = os.path.join(run_spec_dirs, run_spec_dir)
                if not os.path.isdir(full_run_spec_path):
                    continue

                metrics_text: str = Path(os.path.join(full_run_spec_path, "metrics.txt")).read_text()

                if run_spec_dir in ready_run_spec_dir_to_description:
                    run_spec: str = ready_run_spec_dir_to_description[run_spec_dir]
                    ready_content.append(f"{run_spec} - \n{metrics_text}\n")
                else:
                    run_spec: str = wip_run_spec_dir_to_description[run_spec_dir]
                    wip_content.append(f"{run_spec} - {metrics_text}")

        # Write out the status page with the WIP RunSpecs first
        with open(self.status_path, "w") as f:
            f.write("\n".join(wip_content))
            f.write("\n" * 2 + "-" * 150 + "\n" * 2)
            f.write("\n".join(ready_content))

    def run_benchmarking(self, run_spec: str, dry_run: bool = False):
        command: List[str] = [
            Runner.BENCHMARK_RUN_EXECUTABLE,
            f"--url {self.url}",
            f"--api-key-path {self.api_key_path}",
            f"--output-path {self.output_path}",
            f"--num-threads {self.num_threads}",
            f"--run-specs {run_spec}",
        ]
        if dry_run:
            command.append("--dry-run")

        try:
            subprocess.check_call(
                " ".join(command), shell=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"There was an error while running the {Runner.BENCHMARK_RUN_EXECUTABLE} command: {e}")
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
