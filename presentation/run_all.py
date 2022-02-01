import argparse
import subprocess
from tqdm import tqdm
from typing import List, Optional

from pyhocon import ConfigFactory, ConfigTree

"""
Script that runs all the RunSpecs in run_specs.conf.

Usage:
    
    python3 presentation/run_all.py --api-key-path <Path to CRFM API key> --output-path <Where to output results>

"""

# For READY RunSpecs, evaluate and generate metrics.
READY_STATE = "READY"

# For WIP RunSpecs, just estimate token usage.
WIP_STATE = "WIP"


class Runner:
    BENCHMARK_RUN_EXECUTABLE = "venv/bin/benchmark-run"

    @staticmethod
    def get_run_spec_description(run_spec_name: str, run_spec_args: Optional[ConfigTree]) -> str:
        """Get full the run spec description by appending args to the RunSpec name."""
        if run_spec_args:
            args_str: str = ",".join([f"{arg}={run_spec_args[arg]}" for arg in run_spec_args])
            return f"{run_spec_name}:{args_str}"
        else:
            return run_spec_name

    def __init__(self, url: str, api_key_path: str, output_path: str, num_threads: int):
        self.url: str = url
        self.api_key_path: str = api_key_path
        self.output_path: str = output_path
        self.num_threads: int = num_threads

    def run(self):
        print("Reading RunSpecs from run_specs.conf...")
        conf = ConfigFactory.parse_file("presentation/run_specs.conf")

        print("Running all RunSpecs...")
        for run_spec_name in tqdm(conf):
            state: str = conf[run_spec_name].state
            run_spec: str = Runner.get_run_spec_description(
                run_spec_name, run_spec_args=conf[run_spec_name].get("args", default=None),
            )

            if state == READY_STATE:
                self.run_benchmarking(run_spec)
            elif state == WIP_STATE:
                self.run_benchmarking(run_spec, dry_run=True)
            else:
                raise ValueError(f"RunSpec {run_spec_name} has an invalid state: {state}")

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
            print(f"There was an error while creating the temp instance: {e}")
            raise


def main():
    runner = Runner(
        url=args.url, api_key_path=args.api_key_path, output_path=args.output_path, num_threads=args.num_threads,
    )
    runner.run()
    print("Done.")


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
        "-o", "--output-path", help="Where to save all the output", default="benchmark_output",
    )
    parser.add_argument(
        "-n", "--num-threads", type=int, help="Max number of threads to make requests", default=5,
    )
    args = parser.parse_args()

    main()
