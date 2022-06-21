import argparse
import dataclasses
import os
import traceback
import yaml
from collections import defaultdict
from pathlib import Path

import dacite
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Set
import json

from benchmark.scenario import TEST_SPLIT
from common.authentication import Authentication
from common.general import parse_hocon, write
from common.hierarchical_logger import hlog, htrack
from benchmark.statistic import Stat
from benchmark.run import Run, run_benchmarking, add_run_args, validate_args, LATEST_SYMLINK
from benchmark.runner import RunSpec
from proxy.remote_service import add_service_args, create_authentication
from proxy.models import ALL_MODELS, Model

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
        exit_on_error: bool,
        priority: Optional[int],
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
        self.exit_on_error: bool = exit_on_error
        self.priority: Optional[int] = priority

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


class Summarizer:
    """ Summarize the benchmark results in json files to be displayed in the UI. """

    DATA_AUGMENTATION_STR: str = "data_augmentation="

    def __init__(self, run_suite_path: str, conf_path: str):
        """ Initialize the summarizer. """
        self.run_suite_path: str = run_suite_path
        self.conf_path: str = conf_path
        self.runs: List[Run] = self.load_runs()

    @staticmethod
    def load_run_spec_json(run_spec_file_path: str) -> RunSpec:
        """ Load a RunSpec object from a json file. """
        with open(run_spec_file_path, "r") as f:
            j = json.load(f)
            return dacite.from_dict(RunSpec, j)

    @staticmethod
    def load_stats_json(stats_file_path: str) -> List[Stat]:
        """ Load stats from a json file. """
        stats: List[Stat] = []
        with open(stats_file_path, "r") as f:
            j = json.load(f)
            for stat in j:
                stat = dacite.from_dict(Stat, stat)
                stats.append(stat)
        return stats

    @staticmethod
    def load_runs_json(runs_file_path: str) -> List[Run]:
        """ Load runs from runs.json. """
        runs: List[Run] = []
        with open(runs_file_path, "r") as f:
            j = json.load(f)
            for r in j:
                run = dacite.from_dict(Run, r)
                runs.append(run)
        return runs

    def load_run_from_directory(self, run_dir: str) -> Run:
        """ Load the run stored in run_dir. """
        run_dict = {
            "run_dir": run_dir,
            "run_spec": self.load_run_spec_json(os.path.join(run_dir, "run_spec.json")),
            "stats": self.load_stats_json(os.path.join(run_dir, "metrics.json")),
        }
        return Run(**run_dict)

    def load_runs(self) -> List[Run]:
        """ Load the corresponding runs for the run specs in run_specs.json. """
        runs: List[Run] = []
        run_specs_path: str = os.path.join(self.run_suite_path, "run_specs.json")

        if os.path.exists(run_specs_path):
            with open(run_specs_path) as f:
                j = json.load(f)
                for rs in j:
                    run_spec = dacite.from_dict(RunSpec, rs)
                    run_dir_path: str = os.path.join(self.run_suite_path, run_spec.name)
                    run_spec_path: str = os.path.join(run_dir_path, "run_spec.json")
                    metrics_path: str = os.path.join(run_dir_path, "metrics.json")

                    if os.path.exists(run_spec_path) and os.path.exists(metrics_path):
                        run = self.load_run_from_directory(run_dir_path)
                        runs.append(run)
        else:
            hlog(f"Summarizer won't run because {run_specs_path} doesn't exist yet. This is expected in a dry run.")
        return runs

    @staticmethod
    def get_scenario_name_from_run_spec(run_spec: RunSpec) -> str:
        """ Get the scenario name from the RunSpec name.

            run_spec.name                                         => scenario name
            'boolq,model=ai21_j1-jumbo'                           => 'boolq'
            'math:subject=prealgebra,level=5,model=ai21_j1-jumbo' => 'math:subject=prealgebra'
        """
        return run_spec.name.split(",")[0]

    @staticmethod
    def filter_runs_by_model_name(runs: List[Run], model_name: str) -> List[Run]:
        return [r for r in runs if r.run_spec.adapter_spec.model == model_name]

    @staticmethod
    def filter_runs_by_scenario_name(runs: List[Run], scenario_name: str) -> List[Run]:
        return [r for r in runs if Summarizer.get_scenario_name_from_run_spec(r.run_spec) == scenario_name]

    @staticmethod
    def filter_runs_with_perturbations(runs: List[Run]) -> List[Run]:
        return [r for r in runs if not r.run_spec.data_augmenter_spec.perturbations]

    @staticmethod
    def filter_stats(
        stats: List[Stat],
        stat_name: str,
        k: Optional[int] = None,
        split: str = TEST_SPLIT,
        perturbation_name: Optional[str] = None,
    ) -> List[Stat]:
        # We perform filters in separate lines to observe how the stats list change with each
        filtered_stats = [s for s in stats if s.name.name == stat_name]
        filtered_stats = [s for s in filtered_stats if s.name.k == k]
        filtered_stats = [s for s in filtered_stats if s.name.split == split]
        filtered_stats = [s for s in filtered_stats if s.name.perturbation.name == perturbation_name]
        return filtered_stats

    @staticmethod
    def get_model_dict(runs: List[Run], model: Model) -> Dict[str, Any]:
        """ Create a model dictionary containing all the information related to a model and its runs. """
        # Initialize a model dict with the fields of a model
        model_dict = vars(model)
        filtered_runs = Summarizer.filter_runs_by_model_name(runs, model.name)
        model_dict["runs"] = [dataclasses.asdict(run) for run in filtered_runs]
        return model_dict

    def write_runs(self):
        write(
            os.path.join(self.run_suite_path, "runs.json"),
            json.dumps(list(map(dataclasses.asdict, self.runs)), indent=2),
        )

    def write_model_stats(self):
        """ Compute model stats and output them to <self.run_suite_path>/model_stats.json """
        # Populate the model stats for each model
        model_stats = []
        for model in ALL_MODELS:
            model_dict = self.get_model_dict(self.runs, model)
            model_stats.append(model_dict)

        # Write the stats
        write(os.path.join(self.run_suite_path, "model_stats.json"), json.dumps(model_stats, indent=2))

    def write_scenario_stats(self):
        """ Computes model stats and output them to <self.run_suite_path>/scenario_stats.json """
        # @TODO Create scenario stats json
        scenario_stats = []

        # Write the stats
        write(
            os.path.join(self.run_suite_path, "scenario_stats.json"), json.dumps(scenario_stats, indent=2),
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
        exit_on_error=args.exit_on_error,
        priority=args.priority,
    )

    # Run the benchmark!
    runner.run()

    # Output JSON files summarizing the benchmark results which will be loaded in the web interface
    summarizer = Summarizer(run_suite_path=os.path.join(args.output_path, "runs", args.suite), conf_path=args.conf_path)
    summarizer.write_runs()
    summarizer.write_model_stats()
    summarizer.write_scenario_stats()

    hlog("Done.")
