import argparse
import dataclasses
import json
import os
import traceback
import yaml
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Set, Dict

from common.authentication import Authentication
from common.general import parse_hocon, write
from common.hierarchical_logger import hlog, htrack
from benchmark.run import run_benchmarking, add_run_args, validate_args, LATEST_SYMLINK
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
        suite: str,
        num_threads: int,
        dry_run: Optional[bool],
        skip_instances: bool,
        max_eval_instances: Optional[int],
        models_to_run: Optional[List[str]],
    ):
        self.auth: Authentication = auth
        self.conf_path: str = conf_path
        self.url: str = url
        self.output_path: str = output_path
        self.suite: str = suite
        self.num_threads: int = num_threads
        self.dry_run: Optional[bool] = dry_run
        self.skip_instances: bool = skip_instances
        self.max_eval_instances: Optional[int] = max_eval_instances
        self.models_to_run: Optional[List[str]] = models_to_run

    @staticmethod
    def update_status_page(output_dir: str, wip_content: List[str], ready_content: List[str]):
        """
        Updates the status page with the WIP and READY `RunSpec`s results.
        """
        status: str = "\n".join(wip_content + ["", "-" * 150, ""] + ready_content)
        write(os.path.join(output_dir, "status.txt"), status)

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
        suite_dir: str = os.path.join(runs_dir, self.suite)
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

            try:
                new_run_specs = run_benchmarking(
                    run_spec_descriptions=[run_spec_description],
                    auth=self.auth,
                    url=self.url,
                    num_threads=self.num_threads,
                    output_path=self.output_path,
                    suite=self.suite,
                    dry_run=dry_run,
                    skip_instances=self.skip_instances,
                    max_eval_instances=self.max_eval_instances,
                    models_to_run=self.models_to_run,
                )
                run_specs.extend(new_run_specs)

                for run_spec in new_run_specs:
                    run_dir: str = os.path.join(suite_dir, run_spec.name)
                    # Get the metric output, so we can display it on the status page
                    metrics_text: str = Path(os.path.join(run_dir, "metrics.txt")).read_text()
                    if status == READY_STATUS:
                        ready_content.append(f"{run_spec} - \n{metrics_text}\n")
                    else:
                        wip_content.append(f"{run_spec} - {metrics_text}")

                    # Keep track of all the names of the metrics that have been computed
                    with open(os.path.join(run_dir, "metrics.json")) as f:
                        for metric in json.load(f):
                            computed_metrics_to_scenarios[metric["name"]["name"]].add(run_spec.name.split(":")[0])

                # Update the status page after processing every `RunSpec` description
                AllRunner.update_status_page(suite_dir, wip_content, ready_content)

            except Exception:
                hlog(f"Error when running {run_spec_description}:\n{traceback.format_exc()}")

        if len(run_specs) == 0:
            hlog("There were no RunSpecs or they got filtered out.")
            return

        # Write out all the `RunSpec`s and models to json files
        write(
            os.path.join(suite_dir, "run_specs.json"), json.dumps(list(map(dataclasses.asdict, run_specs)), indent=2),
        )
        all_models = [dataclasses.asdict(model) for model in ALL_MODELS]
        write(os.path.join(self.output_path, "models.json"), json.dumps(all_models, indent=2))

        # Create a symlink runs/latest -> runs/<name_of_suite>,
        # so runs/latest always points to the latest run suite.
        symlink_path: str = os.path.abspath(os.path.join(runs_dir, LATEST_SYMLINK))
        if os.path.islink(symlink_path):
            # Remove the previous symlink if it exists.
            os.unlink(symlink_path)
        os.symlink(os.path.abspath(suite_dir), symlink_path)

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
        "--models-to-run",
        nargs="+",
        help="Only RunSpecs with these models specified. If no model is specified, run everything.",
        default=None,
    )
    add_run_args(parser)
    args = parser.parse_args()
    validate_args(args)

    runner = AllRunner(
        # The benchmarking framework will not make any requests to the proxy server when
        # `skip_instances` is set. In that case, just pass in a dummy API key.
        auth=Authentication("test") if args.skip_instances else create_authentication(args),
        conf_path=args.conf_path,
        url=args.server_url,
        output_path=args.output_path,
        suite=args.suite,
        num_threads=args.num_threads,
        dry_run=args.dry_run,
        skip_instances=args.skip_instances,
        max_eval_instances=args.max_eval_instances,
        models_to_run=args.models_to_run,
    )
    runner.run()
    hlog("Done.")
