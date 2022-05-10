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
from common.statistic import Stat
from benchmark.augmentations.data_augmenter import DataAugmenterSpec
from benchmark.augmentations.perturbation_description import PerturbationDescription
from benchmark.adapter import AdapterSpec
from benchmark.run import Run, run_benchmarking, add_run_args
from benchmark.runner import RunSpec
from benchmark.metric import MetricName, MetricSpec
from benchmark.scenario import ScenarioSpec
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

            # Update the status page after processing every `RunSpec` description
            self._update_status_page(wip_content, ready_content)

        # Write out all the `RunSpec`s and models to json files
        write(
            os.path.join(self.output_path, "run_specs.json"),
            json.dumps(list(map(dataclasses.asdict, run_specs)), indent=2),
        )
        all_models = [dataclasses.asdict(model) for model in ALL_MODELS]
        write(os.path.join(self.output_path, "models.json"), json.dumps(all_models, indent=2))

    def _update_status_page(self, wip_content: List[str], ready_content: List[str]):
        """
        Updates the status page with the WIP and READY `RunSpec`s results.
        """
        status: str = "\n".join(wip_content + ["", "-" * 150, ""] + ready_content)
        write(os.path.join(self.output_path, "status.txt"), status)


class Summarizer:
    """ Summarizes the results. """

    DATA_AUGMENTATION_STR: str = "data_augmentation="

    def __init__(
        self, output_path: str,
    ):
        self.output_path: str = output_path
        self.runs: List[Run] = self._load_runs()

    @staticmethod
    def _load_run_spec(run_dir: str) -> RunSpec:
        run_spec_path = os.path.join(run_dir, "run_spec.json")
        with open(run_spec_path, "r") as f:
            j = json.load(f)
            run_spec_dict = {
                "name": j["name"],
                "scenario": ScenarioSpec(**j["scenario"]),
                "adapter_spec": AdapterSpec(**j["adapter_spec"]),
                "metrics": [MetricSpec(**m) for m in j["metrics"]],
                "data_augmenter_spec": DataAugmenterSpec(**j["data_augmenter_spec"]),
            }
        return RunSpec(**run_spec_dict)

    @staticmethod
    def _load_metrics(run_dir: str) -> List[Stat]:
        metrics: List[Stat] = []
        with open(os.path.join(run_dir, "metrics.json"), "r") as f:
            j = json.load(f)
            for m in j:
                m["name"]["perturbation"] = PerturbationDescription(m["name"]["perturbation"])
                m["name"] = MetricName(**m["name"])
                stat = Stat(name=m["name"])
                # Stat constructor doesn't accept default values for the
                # following, so we set them manually
                stat.count = m["count"]
                stat.sum = m["sum"]
                stat.sum_squared = m["sum_squared"] if "sum_squared" in m else 0
                stat.min = m["min"]
                stat.max = m["max"]
                stat.values = m["values"]
                metrics.append(stat)
        return metrics

    def _load_run(self, runs_path: str, dir_name: str) -> Run:
        run_dir = os.path.join(runs_path, dir_name)
        run_dict = {
            "directory_name": os.path.join(runs_path, dir_name),
            "run_spec": self._load_run_spec(run_dir),
            "metrics": self._load_metrics(run_dir),
        }
        return Run(**run_dict)

    def _load_runs(self) -> List[Run]:
        runs = []
        runs_path = os.path.join(self.output_path, "runs")
        for dir_name in os.listdir(runs_path):
            # Filter out hidden directories, which start with .
            if dir_name[0] != ".":
                # Only load a run if it has a run_spec.json
                run_spec_path = os.path.join(runs_path, dir_name, "run_spec.json")
                if os.path.exists(run_spec_path):
                    run = self._load_run(runs_path, dir_name)
                    runs.append(run)
        return runs

    def summarize_model_stats(self):
        """ Computes scenario stats and output them <self.output_path>/model_stats.json """
        # @TODO Create model stats json
        model_stats = {}

        # Write the stats
        with open(os.path.join(self.output_path, "model_stats.json"), "w") as f:
            json.dump(model_stats, f, indent=2)

    def summarize_scenario_stats(self):
        """ Computes scenario stats and output them <self.output_path>/scenario_stats.json """
        # @TODO Create scenario stats json
        scenario_stats = {}

        # Write the stats
        with open(os.path.join(self.output_path, "scenario_stats.json"), "w") as f:
            json.dump(scenario_stats, f, indent=2)


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
    # Run the benchmark!
    # @TODO Commenting out to test summarizer, remove before merging
    # runner.run()
    # Summarize the results for use in the UI
    summarizer = Summarizer(output_path=args.output_path)
    summarizer.summarize_model_stats()
    summarizer.summarize_scenario_stats()
    hlog("Done.")
