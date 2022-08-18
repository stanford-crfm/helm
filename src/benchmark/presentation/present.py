import argparse
import dataclasses
import os
import traceback

from tqdm import tqdm
from typing import List, Optional
import json

from common.authentication import Authentication
from common.general import parse_hocon, write
from common.hierarchical_logger import hlog, htrack
from benchmark.run import run_benchmarking, add_run_args, validate_args, LATEST_SYMLINK
from benchmark.runner import RunSpec
from proxy.services.remote_service import add_service_args, create_authentication

"""
Runs all the RunSpecs in run_specs.conf and outputs JSON files.
TODO: rename this file to `run_all.py`

Usage:

    venv/bin/benchmark-present

"""

READY_STATUS = "READY"
WIP_STATUS = "WIP"


class AllRunner:
    """Runs all RunSpecs specified in the configuration file."""

    def __init__(
        self,
        auth: Authentication,
        conf_path: str,
        url: str,
        local: bool,
        local_path: str,
        output_path: str,
        suite: str,
        num_threads: int,
        dry_run: Optional[bool],
        skip_instances: bool,
        max_eval_instances: Optional[int],
        models_to_run: Optional[List[str]],
        scenario_groups_to_run: Optional[List[str]],
        exit_on_error: bool,
        priority: Optional[int],
    ):
        self.auth: Authentication = auth
        self.conf_path: str = conf_path
        self.url: str = url
        self.local: bool = local
        self.local_path: str = local_path
        self.output_path: str = output_path
        self.suite: str = suite
        self.num_threads: int = num_threads
        self.dry_run: Optional[bool] = dry_run
        self.skip_instances: bool = skip_instances
        self.max_eval_instances: Optional[int] = max_eval_instances
        self.models_to_run: Optional[List[str]] = models_to_run
        self.scenario_groups_to_run: Optional[List[str]] = scenario_groups_to_run
        self.exit_on_error: bool = exit_on_error
        self.priority: Optional[int] = priority

    @htrack(None)
    def run(self):
        hlog("Reading RunSpecs from run_specs.conf...")
        with open(self.conf_path) as f:
            conf = parse_hocon(f.read())

        hlog("Running all RunSpecs...")
        run_specs: List[RunSpec] = []
        runs_dir: str = os.path.join(self.output_path, "runs")
        suite_dir: str = os.path.join(runs_dir, self.suite)

        for run_spec_description, run_spec_state in tqdm(conf.items()):
            # We placed double quotes around the descriptions since they can have colons or equal signs.
            # There is a bug with pyhocon. pyhocon keeps the double quote when there is a ".", ":" or "=" in the string:
            # https://github.com/chimpler/pyhocon/issues/267
            # We have to manually remove the double quotes from the descriptions.
            run_spec_description = run_spec_description.replace('"', "")
            status: str = run_spec_state.status
            if status not in (READY_STATUS, WIP_STATUS):
                raise ValueError(f"RunSpec {run_spec_description} has an invalid status: {status}")

            priority: int = run_spec_state.priority
            if self.priority is not None and priority > self.priority:
                hlog(f"Skipping {run_spec_description} since its priority {priority} > {self.priority}")
                continue

            # Use `dry_run` flag if set, else use what's in the file.
            dry_run = self.dry_run if self.dry_run is not None else status == WIP_STATUS

            try:
                new_run_specs = run_benchmarking(
                    run_spec_descriptions=[run_spec_description],
                    auth=self.auth,
                    url=self.url,
                    local=self.local,
                    local_path=self.local_path,
                    num_threads=self.num_threads,
                    output_path=self.output_path,
                    suite=self.suite,
                    dry_run=dry_run,
                    skip_instances=self.skip_instances,
                    max_eval_instances=self.max_eval_instances,
                    models_to_run=self.models_to_run,
                    scenario_groups_to_run=self.scenario_groups_to_run,
                )
                run_specs.extend(new_run_specs)

            except Exception as e:
                if self.exit_on_error:
                    raise e
                else:
                    hlog(f"Error when running {run_spec_description}:\n{traceback.format_exc()}")

        if len(run_specs) == 0:
            hlog("There were no RunSpecs or they got filtered out.")
            return

        # Write out all the `RunSpec`s and models to json files
        write(
            os.path.join(suite_dir, "run_specs.json"), json.dumps(list(map(dataclasses.asdict, run_specs)), indent=2),
        )

        # Print out all the models and scenario groups available
        # This makes it easy to select individual ones to run.
        models = set()
        groups = set()
        for run_spec in run_specs:
            models.add(run_spec.adapter_spec.model)
            for group in run_spec.groups:
                groups.add(group)
        hlog(f"{len(models)} models: {' '.join(models)}")
        hlog(f"{len(groups)} scenario groups: {' '.join(groups)}")

        # Create a symlink runs/latest -> runs/<name_of_suite>,
        # so runs/latest always points to the latest run suite.
        symlink_path: str = os.path.abspath(os.path.join(runs_dir, LATEST_SYMLINK))
        if os.path.islink(symlink_path):
            # Remove the previous symlink if it exists.
            os.unlink(symlink_path)
        os.symlink(os.path.abspath(suite_dir), symlink_path)


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
        help="Only RunSpecs with these models specified. If no model is specified, runs with all models.",
        default=None,
    )
    parser.add_argument(
        "--scenario-groups-to-run",
        nargs="+",
        help="Only RunSpecs with these scenario groups specified. If no scenario group is specified, runs with all models.",
        default=None,
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        default=None,
        help="Fail and exit immediately if a particular RunSpec fails.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Run RunSpecs with priority less than or equal to this number. "
        "If a value for --priority is not specified, run on everything",
    )
    add_run_args(parser)
    args = parser.parse_args()
    validate_args(args)

    runner = AllRunner(
        # Use a dummy API key when `skip_instances` or `local` is set.
        # The benchmarking framework will not make any requests to the proxy server when
        # `skip_instances` is set, so a valid API key is not necessary.
        # Setting `local` will run and cache everything locally.
        auth=Authentication("") if args.skip_instances or args.local else create_authentication(args),
        conf_path=args.conf_path,
        url=args.server_url,
        local=args.local,
        local_path=args.local_path,
        output_path=args.output_path,
        suite=args.suite,
        num_threads=args.num_threads,
        dry_run=args.dry_run,
        skip_instances=args.skip_instances,
        max_eval_instances=args.max_eval_instances,
        models_to_run=args.models_to_run,
        scenario_groups_to_run=args.scenario_groups_to_run,
        exit_on_error=args.exit_on_error,
        priority=args.priority,
    )

    # Run the benchmark!
    runner.run()

    hlog("Done.")
