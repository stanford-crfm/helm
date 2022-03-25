import argparse
import dataclasses
import os.path
from pathlib import Path

from tqdm import tqdm
from typing import List, Optional
import json

from common.authentication import Authentication
from common.general import parse_hocon, write
from common.hierarchical_logger import hlog, htrack
from common.object_spec import parse_object_spec
from benchmark.run_specs import construct_run_specs
from benchmark.run import run_benchmarking, add_run_args
from benchmark.runner import RunSpec
from proxy.remote_service import add_service_args, create_authentication
from proxy.models import ALL_MODELS

"""
Runs all the RunSpecs in run_specs.conf and outputs a status page.

Usage:

    venv/bin/benchmark-present -s <Where to output the status page>

"""

# For READY RunSpecs, evaluate and generate metrics.
READY_STATUS = "READY"

# For WIP RunSpecs, just estimate token usage.
WIP_STATUS = "WIP"


class AllRunner:
    """Runs all RunSpecs specified in the configuration file."""

    def __init__(
        self,
        auth: Authentication,
        conf_path: str,
        url: str,
        output_path: str,
        num_threads: int,
        dry_run: Optional[bool],
        very_dry_run: Optional[bool],
        max_eval_instances: Optional[int],
    ):
        self.auth: Authentication = auth
        self.conf_path: str = conf_path
        self.url: str = url
        self.output_path: str = output_path
        self.num_threads: int = num_threads
        self.dry_run: bool = dry_run
        self.very_dry_run: bool = very_dry_run
        self.max_eval_instances: Optional[int] = max_eval_instances

    @htrack(None)
    def run(self):
        hlog("Reading RunSpecs from run_specs.conf...")
        with open(self.conf_path) as f:
            conf = parse_hocon(f.read())

        if self.very_dry_run:
            hlog("Verifying the run spec descriptions...")
            self.check_run_spec_descriptions(conf.keys())
            return

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
        run_specs: List[RunSpec] = []
        runs_dir: str = os.path.join(self.output_path, "runs")
        for run_spec_description, run_spec_state in tqdm(conf.items()):
            # We placed double quotes around the descriptions since they can have colons or equal signs.
            # There is a bug with pyhocon. pyhocon keeps the double quote when there is a ".", ":" or "=" in the string:
            # https://github.com/chimpler/pyhocon/issues/267
            # We have to manually remove the double quotes from the descriptions.
            run_spec_description = run_spec_description.replace('"', "")
            status: str = run_spec_state.status

            if status != READY_STATUS and status != WIP_STATUS:
                raise ValueError(f"RunSpec {run_spec_description} has an invalid status: {status}")

            # Use `dry_run` flag if set, else use what's in the file.
            dry_run = self.dry_run if self.dry_run is not None else status == WIP_STATUS

            run_benchmarking(
                run_spec_descriptions=[run_spec_description],
                auth=self.auth,
                url=self.url,
                num_threads=self.num_threads,
                output_path=self.output_path,
                dry_run=dry_run,
                max_eval_instances=self.max_eval_instances,
            )

            with open(os.path.join(runs_dir, run_spec_description, "run_spec.json")) as f:
                run_spec = json.load(f)
                run_specs.append(run_spec)

            # Get the metric output, so we can display it on the status page
            metrics_text: str = Path(os.path.join(runs_dir, run_spec_description, "metrics.txt")).read_text()
            if status == READY_STATUS:
                ready_content.append(f"{run_spec} - \n{metrics_text}\n")
            else:
                wip_content.append(f"{run_spec} - {metrics_text}")

        # Write out the status page with the WIP RunSpecs first
        status = "\n".join(wip_content + ["", "-" * 150, ""] + ready_content)
        write(os.path.join(self.output_path, "status.txt"), status)

        write(os.path.join(self.output_path, "run_specs.json"), json.dumps(run_specs, indent=2))

        all_models = [dataclasses.asdict(model) for model in ALL_MODELS]
        write(os.path.join(self.output_path, "models.json"), json.dumps(all_models, indent=2))

    def check_run_spec_descriptions(self, run_spec_descriptions: List[str]):
        """Skips downloading datasets and execution and just ensure run spec descriptions are parsed correctly."""
        for run_spec_description in run_spec_descriptions:
            # We placed double quotes around the descriptions since they can have colons or equal signs.
            # There is a bug with pyhocon. pyhocon keeps the double quote when there is a ".", ":" or "=" in the string:
            # https://github.com/chimpler/pyhocon/issues/267
            # Therefore, we have to manually remove the double quotes from the descriptions.
            run_spec_description = run_spec_description.replace('"', "")
            construct_run_specs(parse_object_spec(run_spec_description))


def main():
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument(
        "-c",
        "--conf-path",
        help="Where to read RunSpecs to run from",
        default="src/benchmark/presentation/run_specs.conf",
    )
    parser.add_argument(
        "--very-dry-run",
        action="store_true",
        help="Skips downloading datasets and execution and just ensure run spec descriptions are parsed correctly.",
    )
    add_run_args(parser)
    args = parser.parse_args()
    runner = AllRunner(
        # The benchmarking framework will not make any requests to the proxy server when
        # `dry_run` or `very_dry_run` is set. In that case, just pass in a dummy API key.
        auth=Authentication("test") if args.dry_run or args.very_dry_run else create_authentication(args),
        conf_path=args.conf_path,
        url=args.server_url,
        output_path=args.output_path,
        num_threads=args.num_threads,
        dry_run=args.dry_run,
        very_dry_run=args.very_dry_run,
        max_eval_instances=args.max_eval_instances,
    )
    runner.run()
    hlog("Done.")
