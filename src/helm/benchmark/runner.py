import os
import typing
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

from tqdm import tqdm

from helm.common.general import ensure_directory_exists, write
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.cache import cache_stats
from helm.common.codec import from_json, to_json
from .augmentations.data_augmenter import DataAugmenterSpec
from .scenarios.scenario import Scenario, ScenarioSpec, create_scenario, Instance, with_instance_ids
from .adaptation.adapters.adapter import Adapter
from .adaptation.adapters.adapter_factory import AdapterFactory
from .adaptation.scenario_state import ScenarioState
from .adaptation.adapter_spec import AdapterSpec
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor
from .metrics.metric_name import MetricName
from .metrics.metric_service import MetricService
from .metrics.metric import Metric, MetricSpec, MetricResult, PerInstanceStats, create_metric, Stat
from .metrics.tokens_metric import TokensMetric
from .window_services.tokenizer_service import TokenizerService


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
        cache_step_results: bool = True,
    ):
        self.executor = Executor(execution_spec)
        self.dry_run: bool = execution_spec.dry_run
        self.tokenizer_service = TokenizerService(self.executor.service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.service, execution_spec.auth)
        self.skip_instances: bool = skip_instances
        self.cache_step_results: bool = cache_step_results

        ensure_directory_exists(output_path)
        # Decide where to save the raw data (e.g., "output/scenarios/mmlu").
        self.scenarios_path: str = os.path.join(output_path, "scenarios")
        ensure_directory_exists(self.scenarios_path)

        # Output the results under a folder with the name of the suite
        self.runs_path: str = os.path.join(output_path, "runs", suite)

        # The path where to cache files needs to compute metrics, e.g., human evaluation results
        self.eval_cache_path: str = os.path.join(self.runs_path, "eval_cache")
        ensure_directory_exists(self.eval_cache_path)

    def run_all(self, run_specs: List[RunSpec]):
        for run_spec in tqdm(run_specs):
            with htrack_block(f"Running {run_spec.name}"):
                self.run_one(run_spec)

    def _json_file_name(self, run_spec: RunSpec, file_name: str) -> str:
        return os.path.join(self.runs_path, run_spec.name, file_name)

    def _scenario_step(self, run_spec: RunSpec) -> Scenario:
        scenario: Scenario = create_scenario(run_spec.scenario_spec)
        write(self._json_file_name(run_spec, "scenario.json"), to_json(scenario))
        return scenario

    def _instances_step(self, run_spec: RunSpec, scenario: Scenario) -> List[Instance]:
        instances_path: str = self._json_file_name(run_spec, "instances.json")
        if self.cache_step_results and os.path.exists(instances_path):
            with open(instances_path, "r") as f:
                hlog(f"Using cached results from {instances_path} and skipping step")
                return from_json(f.read(), List[Instance])
        instances: List[Instance]
        if not self.skip_instances:
            # This `output_path` will be used when `Adapter` calls `Scenario.get_instances`.
            scenario.output_path = os.path.join(self.scenarios_path, scenario.name)
            ensure_directory_exists(scenario.output_path)

            # Create the instances of the scenario
            with htrack_block("scenario.get_instances"):
                instances = scenario.get_instances()

            # Give each instance a unique ID
            instances = with_instance_ids(instances)

            # Get the instances necessary for this run.
            adapter: Adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, self.tokenizer_service)
            instances = adapter.get_run_instances(instances)

            # Data preprocessing
            instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
                instances, self.executor.execution_spec.parallelism
            )
        else:
            instances = []
        write(instances_path, to_json(instances))
        return instances

    def _scenario_state_step(self, run_spec: RunSpec, instances: List[Instance]) -> ScenarioState:
        scenario_state_path: str = self._json_file_name(run_spec, "scenario_state.json")
        if self.cache_step_results and os.path.exists(scenario_state_path):
            with open(scenario_state_path, "r") as f:
                hlog(f"Using cached results from {scenario_state_path} and skipping step")
                return from_json(f.read(), ScenarioState)

        # Adapt (convert to requests)
        adapter: Adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, self.tokenizer_service)
        scenario_state: ScenarioState = adapter.adapt(instances, self.executor.execution_spec.parallelism)

        # Execute (fill up results)
        scenario_state = self.executor.execute(scenario_state)
        write(scenario_state_path, to_json(scenario_state))
        return scenario_state

    def _metrics_step(
        self, run_spec: RunSpec, scenario_state: ScenarioState
    ) -> Tuple[List[Stat], List[PerInstanceStats]]:
        # Apply the metrics
        # When performing a dry run, only estimate the number of tokens instead
        # of calculating the metrics.
        metrics: List[Metric] = (
            [] if self.dry_run else [create_metric(metric_spec) for metric_spec in run_spec.metric_specs]
        ) + [TokensMetric()]
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
        write(self._json_file_name(run_spec, "stats.json"), to_json(stats))
        write(self._json_file_name(run_spec, "per_instance_stats.json"), to_json(per_instance_stats))
        return (stats, per_instance_stats)

    def run_one(self, run_spec: RunSpec):
        run_path: str = os.path.join(self.runs_path, run_spec.name)
        ensure_directory_exists(run_path)

        scenario: Scenario = self._scenario_step(run_spec)
        instances: List[Instance] = self._instances_step(run_spec, scenario)
        scenario_state: ScenarioState = self._scenario_state_step(run_spec, instances)
        self._metrics_step(run_spec, scenario_state)

        cache_stats.print_status()
