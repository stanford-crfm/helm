import argparse
import os.path
from pathlib import Path

from tqdm import tqdm
from typing import List

from common.authentication import Authentication
from common.general import parse_hocon
from common.hierarchical_logger import hlog
from benchmark.run import run_benchmarking
from proxy.remote_service import add_service_args, create_authentication

"""
Script that runs all the RunSpecs in run_specs.conf and outputs a status page.

Usage:
    
    python3 presentation/run_all.py -s <Where to output the status page>

"""

# For READY RunSpecs, evaluate and generate metrics.
READY_STATUS = "READY"

# For WIP RunSpecs, just estimate token usage.
WIP_STATUS = "WIP"


class AllRunner:
    """Runs all RunSpecs specified in the configuration file."""

    def __init__(
        self, auth: Authentication, conf_path: str, url: str, output_path: str, status_path: str, num_threads: int
    ):
        self.auth: Authentication = auth
        self.conf_path: str = conf_path
        self.url: str = url
        self.output_path: str = output_path
        self.status_path: str = status_path
        self.num_threads: int = num_threads

    def run(self):
        hlog("Reading RunSpecs from run_specs.conf...")
        with open(self.conf_path) as f:
            conf = parse_hocon(f.read())

        # Keep track of the output of READY and WIP RunSpecs separately
        ready_content: List[str] = ["##### Ready Run Specs ######\n"]
        wip_content: List[str] = [
            "##### Work In Progress Run Specs #####",
            "",
            "The following RunSpecs are not ready for evaluation. Instead of evaluating, "
            "we just estimate the number of tokens needed.",
            "",
        ]

        hlog("Running all RunSpecs...")
        runs_dir: str = os.path.join(self.output_path, "runs")
        for run_spec, run_spec_state in tqdm(conf.items()):
            # We placed double quotes around the descriptions since they can have colons.
            # Remove the double quotes from the descriptions.
            run_spec = run_spec.replace('"', "")
            status: str = run_spec_state.status

            if status != READY_STATUS and status != WIP_STATUS:
                raise ValueError(f"RunSpec {run_spec} has an invalid status: {status}")

            run_benchmarking(
                run_spec_descriptions=[run_spec],
                auth=self.auth,
                url=self.url,
                num_threads=self.num_threads,
                output_path=self.output_path,
                dry_run=status == WIP_STATUS,
            )

            # After running the RunSpec, get the metric output, so we can display it on the status page
            metrics_text: str = Path(os.path.join(runs_dir, run_spec, "metrics.txt")).read_text()
            if status == READY_STATUS:
                ready_content.append(f"{run_spec} - \n{metrics_text}\n")
            else:
                wip_content.append(f"{run_spec} - {metrics_text}")

        # Write out the status page with the WIP RunSpecs first
        with open(self.status_path, "w") as f:
            f.write("\n".join(wip_content))
            f.write("\n" * 2 + "-" * 150 + "\n" * 2)
            f.write("\n".join(ready_content))


def main():
    runner = AllRunner(
        auth=create_authentication(args),
        conf_path=args.conf_path,
        url=args.server_url,
        output_path=args.output_path,
        status_path=args.status_path,
        num_threads=args.num_threads,
    )
    runner.run()
    hlog("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument(
        "--conf-path", help="Where to read RunSpecs to run from", default="presentation/run_specs.conf",
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
