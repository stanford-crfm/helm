import dacite
import json
import os
import traceback
import typing
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tqdm import tqdm

from helm.common.general import ensure_directory_exists, write, asdict_without_nones
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.cache import cache_stats
from .augmentations.data_augmenter import DataAugmenterSpec
from .scenarios.scenario import Scenario, ScenarioSpec, create_scenario, Instance, with_instance_ids
from .adaptation.adapters.adapter import Adapter
from .adaptation.adapters.adapter_factory import AdapterFactory
from .adaptation.scenario_state import ScenarioState
from .adaptation.adapter_spec import AdapterSpec
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor
from .metrics.dry_run_metrics import DryRunMetric
from .metrics.metric_name import MetricName
from .metrics.metric_service import MetricService
from .metrics.metric import Metric, MetricSpec, MetricResult, PerInstanceStats, create_metric, Stat
from .window_services.tokenizer_service import TokenizerService


LATEST_SYMLINK: str = "latest"


class RunnerError(Exception):
    """Error that happens in the Runner."""

    pass


@dataclass(frozen=True)
class RunSpec:
    """
    Specifies how to do a single run, which gets a scenario, adapts it, and
    computes a list of stats based on the defined metrics.
    """

    # Unique identifier of the RunSpec
    name: str

    # Which scenario
    scenario_spec: ScenarioSpec

    # Specifies how to adapt an instance into a set of requests
    adapter_spec: AdapterSpec

    # What to evaluate on
    metric_specs: List[MetricSpec]

    # Data augmenter. The default `DataAugmenterSpec` does nothing.
    data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec()

    # Groups that this run spec belongs to (for aggregation)
    groups: List[str] = field(default_factory=list)

    def __post_init__(self):
        """
        `self.name` is used as the name of the output folder for the `RunSpec`.
        Clean up `self.name` by replacing any "/"'s with "_".
        """
        # TODO: Don't mutate name! clean this up before passing it into the constructor here
        object.__setattr__(self, "name", self.name.replace(os.path.sep, "_"))


class Runner:
    """
    The main entry point for running the entire benchmark.  Mostly just
    dispatches to other classes.
    """

    def __init__(
        self,
        execution_spec: ExecutionSpec,
        output_path: str,
        suite: str,
        skip_instances: bool,
        cache_instances: bool,
        cache_instances_only: bool,
        skip_completed_runs: bool,
        exit_on_error: bool,
    ):
        self.executor = Executor(execution_spec)
        self.dry_run: bool = execution_spec.dry_run
        self.tokenizer_service = TokenizerService(self.executor.service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.service, execution_spec.auth)
        self.skip_instances: bool = skip_instances
        self.cache_instances: bool = cache_instances
        self.cache_instances_only: bool = cache_instances_only
        self.skip_completed_runs: bool = skip_completed_runs
        self.exit_on_error: bool = exit_on_error

        ensure_directory_exists(output_path)
        # Decide where to save the raw data (e.g., "output/scenarios/mmlu").
        self.scenarios_path: str = os.path.join(output_path, "scenarios")
        ensure_directory_exists(self.scenarios_path)
        # Decide where to save input instances
        self.instances_path: str = os.path.join(output_path, "scenario_instances")
        ensure_directory_exists(self.instances_path)

        # Output the results under a folder with the name of the suite
        self.runs_path: str = os.path.join(output_path, "runs", suite)

        # The path where to cache files needs to compute metrics, e.g., human evaluation results
        self.eval_cache_path: str = os.path.join(self.runs_path, "eval_cache")
        ensure_directory_exists(self.eval_cache_path)

    def run_all(self, run_specs: List[RunSpec]):
        failed_run_specs: List[RunSpec] = []

        for run_spec in tqdm(run_specs, disable=None):
            try:
                with htrack_block(f"Running {run_spec.name}"):
                    self.run_one(run_spec)
            except Exception as e:
                if self.exit_on_error:
                    raise e
                else:
                    hlog(f"Error when running {run_spec.name}:\n{traceback.format_exc()}")
                    failed_run_specs.append(run_spec)
        if not self.exit_on_error and failed_run_specs:
            failed_runs_str = ", ".join([f'"{run_spec.name}"' for run_spec in failed_run_specs])
            raise RunnerError(f"Failed runs: [{failed_runs_str}]")

    def run_one(self, run_spec: RunSpec):
        # Load the scenario
        scenario: Scenario = create_scenario(run_spec.scenario_spec)

        # This `output_path` will be used when `Adapter` calls `Scenario.get_instances`.
        scenario.output_path = os.path.join(self.scenarios_path, scenario.name)
        ensure_directory_exists(scenario.output_path)

        # This 'output_path' will be used when the model's input instances are saved.
        args_str = ",".join([f"{k}={v}" for k, v in sorted(run_spec.scenario_spec.args.items())])
        scenario_name_with_args = f"{scenario.name}:{args_str}" if args_str else f"{scenario.name}"
        input_instances_output_path = os.path.join(self.instances_path, scenario_name_with_args)
        input_instances_file_path = os.path.join(input_instances_output_path, "input_instances.json")

        run_path: str = os.path.join(self.runs_path, run_spec.name)
        ensure_directory_exists(run_path)

        if self.skip_completed_runs and os.path.exists(os.path.join(run_path, "scenario_state.json")):
            # If scenario_state.json exists, assume that all other output files exist
            # because scenario_state.json is the last output file to be written.
            hlog(f"Skipping run {run_spec.name} because run is completed and all output files exist.")
            return

        # Fetch and initialize the Adapter based on the `AdapterSpec`.
        adapter: Adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, self.tokenizer_service)

        instances: List[Instance]
        if self.skip_instances:
            instances = []
        else:
            if self.cache_instances and os.path.exists(input_instances_file_path):
                with open(input_instances_file_path) as f:
                    json_instances: List[Dict[str, Any]] = json.load(f)
                instances = [dacite.from_dict(Instance, instance) for instance in json_instances]
            else:
                # Create the instances of the scenario
                with htrack_block("scenario.get_instances"):
                    instances = scenario.get_instances()
        if self.cache_instances and not os.path.exists(input_instances_file_path):
            # Save instances to file
            ensure_directory_exists(input_instances_output_path)
            write(
                os.path.join(input_instances_file_path),
                json.dumps([asdict_without_nones(instance) for instance in instances], indent=2),
            )
        if self.cache_instances_only:
            return  # Exit after saving the instances.

        # Give each instance a unique ID
        instances = with_instance_ids(instances)

        # Get the instances necessary for this run.
        instances = adapter.get_run_instances(instances)

        # Data preprocessing
        instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
            instances, self.executor.execution_spec.parallelism
        )

        # Adapt (convert to requests)
        scenario_state: ScenarioState = adapter.adapt(instances, self.executor.execution_spec.parallelism)

        # Execute (fill up results)
        scenario_state = self.executor.execute(scenario_state)

        # Apply the metrics
        # When performing a dry run, only estimate the number of tokens instead
        # of calculating the metrics.
        metrics: List[Metric] = (
            [DryRunMetric()] if self.dry_run else [create_metric(metric_spec) for metric_spec in run_spec.metric_specs]
        )
        stats: List[Stat] = []
        per_instance_stats: List[PerInstanceStats] = []
        with htrack_block(f"{len(metrics)} metrics"):
            for metric in metrics:
                with htrack_block(metric):
                    metric_result: MetricResult = metric.evaluate(
                        scenario_state,
                        self.metric_service,
                        self.eval_cache_path,
                        self.executor.execution_spec.parallelism,
                    )
                    stats.extend(metric_result.aggregated_stats)
                    per_instance_stats.extend(metric_result.per_instance_stats)

        # Check that there aren't duplicate `Stat`s
        # Note: doesn't catch near misses.
        metric_counts: typing.Counter[MetricName] = Counter([stat.name for stat in stats])
        for metric_name, count in metric_counts.items():
            if count > 1:
                hlog(f"WARNING: duplicate metric name {metric_name}")

        # Print out the number of stats
        hlog(f"Generated {len(stats)} stats.")

        if self.skip_instances:
            hlog("skip_instances was True. Skipping writing results out.")
            return

        # Output benchmarking information and results to files
        write(os.path.join(run_path, "run_spec.json"), json.dumps(asdict_without_nones(run_spec), indent=2))

        # Write out scenario
        write(os.path.join(run_path, "scenario.json"), json.dumps(asdict_without_nones(scenario), indent=2))

        # Write scenario state
        write(os.path.join(run_path, "scenario_state.json"), json.dumps(asdict_without_nones(scenario_state), indent=2))

        write(
            os.path.join(run_path, "stats.json"), json.dumps([asdict_without_nones(stat) for stat in stats], indent=2)
        )
        write(
            os.path.join(run_path, "per_instance_stats.json"),
            json.dumps(list(map(asdict_without_nones, per_instance_stats)), indent=2),
        )

        cache_stats.print_status()
