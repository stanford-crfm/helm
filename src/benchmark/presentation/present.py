import argparse
import dataclasses
import os.path
from collections import defaultdict
from pathlib import Path

import json
import yaml
from tqdm import tqdm
from typing import List, Optional, Set, Dict

from common.authentication import Authentication
from common.general import parse_hocon, write
from common.hierarchical_logger import hlog, htrack
from benchmark.run import run_benchmarking, add_run_args
from benchmark.runner import RunSpec
from proxy.remote_service import add_service_args, create_authentication
from proxy.models import ALL_MODELS

"""
Runs all the RunSpecs in run_specs.conf and outputs a status page.

Usage:

    venv/bin/benchmark-present

"""

# For READY RunSpecs, evaluate and generate metrics.
READY_STATUS = "READY"

# For WIP RunSpecs, just estimate token usage.
WIP_STATUS = "WIP"

SCHEMA_YAML_PATH: str = "src/proxy/static/schema.yaml"


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
        skip_instances: bool,
        max_eval_instances: Optional[int],
    ):
        self.auth: Authentication = auth
        self.conf_path: str = conf_path
        self.url: str = url
        self.output_path: str = output_path
        self.num_threads: int = num_threads
        self.dry_run = dry_run
        self.skip_instances = skip_instances
        self.max_eval_instances = max_eval_instances

    @htrack(None)
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
        run_specs: List[RunSpec] = []
        runs_dir: str = os.path.join(self.output_path, "runs")
        computed_metrics_to_scenarios: Dict[str, Set[str]] = defaultdict(set)

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

            new_run_specs = run_benchmarking(
                run_spec_descriptions=[run_spec_description],
                auth=self.auth,
                url=self.url,
                num_threads=self.num_threads,
                output_path=self.output_path,
                dry_run=dry_run,
                skip_instances=self.skip_instances,
                max_eval_instances=self.max_eval_instances,
            )
            run_specs.extend(new_run_specs)

            for run_spec in new_run_specs:
                # Get the metric output, so we can display it on the status page
                metrics_text: str = Path(os.path.join(runs_dir, run_spec.name, "metrics.txt")).read_text()
                if status == READY_STATUS:
                    ready_content.append(f"{run_spec} - \n{metrics_text}\n")
                else:
                    wip_content.append(f"{run_spec} - {metrics_text}")

                # Keep track of all the names of the metrics that have been computed
                with open(os.path.join(runs_dir, run_spec.name, "metrics.json")) as f:
                    for metric in json.load(f):
                        computed_metrics_to_scenarios[metric["name"]["name"]].add(run_spec.name.split(":")[0])

            # Update the status page after processing every `RunSpec` description
            self._update_status_page(wip_content, ready_content)

        # Write out all the `RunSpec`s and models to json files
        write(
            os.path.join(self.output_path, "run_specs.json"),
            json.dumps(list(map(dataclasses.asdict, run_specs)), indent=2),
        )
        all_models = [dataclasses.asdict(model) for model in ALL_MODELS]
        write(os.path.join(self.output_path, "models.json"), json.dumps(all_models, indent=2))

        # Print a warning that list the metrics that do not have a entry in schema.yaml
        metrics_with_entries: Set[str] = set(
            metric_entry["name"] for metric_entry in yaml.safe_load(open(SCHEMA_YAML_PATH))["metrics"]
        )
        missing_metrics_str: str = "\n\t".join(
            [
                f"{metric}: {','.join(scenarios)} "
                for metric, scenarios in computed_metrics_to_scenarios.items()
                if metric not in metrics_with_entries
            ]
        )
        if missing_metrics_str:
            hlog(
                f"WARNING: Missing an entry for the following metrics in {SCHEMA_YAML_PATH}: \n\t{missing_metrics_str}"
            )

    def _update_status_page(self, wip_content: List[str], ready_content: List[str]):
        """
        Updates the status page with the WIP and READY `RunSpec`s results.
        """
        status: str = "\n".join(wip_content + ["", "-" * 150, ""] + ready_content)
        write(os.path.join(self.output_path, "status.txt"), status)


def main():
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument(
        "-c",
        "--conf-path",
        help="Where to read RunSpecs to run from",
        default="src/benchmark/presentation/run_specs.conf",
    )
    add_run_args(parser)
    args = parser.parse_args()

    runner = AllRunner(
        # The benchmarking framework will not make any requests to the proxy server when
        # `skip_instances` is set. In that case, just pass in a dummy API key.
        auth=Authentication("test") if args.skip_instances else create_authentication(args),
        conf_path=args.conf_path,
        url=args.server_url,
        output_path=args.output_path,
        num_threads=args.num_threads,
        dry_run=args.dry_run,
        skip_instances=args.skip_instances,
        max_eval_instances=args.max_eval_instances,
    )
    runner.run()
    hlog("Done.")
