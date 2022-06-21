import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List

from common.general import ensure_directory_exists, write, write_lines
from common.hierarchical_logger import hlog, htrack_block
from .adapter_service import AdapterService
from .augmentations.data_augmenter import DataAugmenterSpec
from .metric_service import MetricService
from .scenario import Scenario, ScenarioSpec, create_scenario, Instance
from .adapter import AdapterSpec, Adapter, ScenarioState
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor
from .metric import Metric, MetricSpec, MetricResult, PerInstanceStatsKey, create_metric, Stat
from .tokens_metric import TokensMetric


@dataclass(frozen=True)
class RunSpec:
    """
    Specifies how to do a single run, which gets a scenario, adapts it, and
    computes a list of metrics.
    """

    # Unique identifier of the RunSpec
    name: str

    # Which scenario
    scenario: ScenarioSpec

    # Specifies how to adapt an instance into a set of requests
    adapter_spec: AdapterSpec

    # What to evaluate on
    metrics: List[MetricSpec]

    # Data augmenter. The default `DataAugmenterSpec` does nothing.
    data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec()


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
        run_specs: List[RunSpec],
        skip_instances: bool,
    ):
        self.executor = Executor(execution_spec)
        self.dry_run: bool = execution_spec.dry_run
        self.adapter_service = AdapterService(self.executor.remote_service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.remote_service, execution_spec.auth)
        self.run_specs: List[RunSpec] = run_specs
        self.skip_instances: bool = skip_instances

        ensure_directory_exists(output_path)
        # Decide where to save the raw data (e.g., "output/scenarios/mmlu").
        self.scenarios_path: str = os.path.join(output_path, "scenarios")
        ensure_directory_exists(self.scenarios_path)

        # Output the results under a folder with the name of the suite
        self.runs_path: str = os.path.join(output_path, "runs", suite)

        # The path where to cache files needs to compute metrics, e.g., human evaluation results
        self.eval_cache_path: str = os.path.join(self.runs_path, "eval_cache")
        ensure_directory_exists(self.eval_cache_path)

    def run_all(self):
        for run_spec in self.run_specs:
            with htrack_block(f"Running {run_spec.name}"):
                self.run_one(run_spec)

    def run_one(self, run_spec: RunSpec):
        # Load the scenario
        scenario: Scenario = create_scenario(run_spec.scenario)

        # This `output_path` will be used when `Adapter` calls `Scenario.get_instances`.
        scenario.output_path = os.path.join(self.scenarios_path, scenario.name)
        ensure_directory_exists(scenario.output_path)
        scenario.definition_path = scenario.get_definition_path()
        run_path: str = os.path.join(self.runs_path, run_spec.name)
        ensure_directory_exists(run_path)

        # Data preprocessing
        if not self.skip_instances:
            instances: List[Instance] = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(scenario)
        else:
            instances = []

        # Adaptation
        adapter = Adapter(run_spec.adapter_spec, self.adapter_service)
        scenario_state: ScenarioState = adapter.adapt(instances)

        # Execution
        scenario_state = self.executor.execute(scenario_state)

        # Apply the metrics
        # When performing a dry run, only estimate the number of tokens instead
        # of calculating the metrics.
        metrics: List[Metric] = ([] if self.dry_run else [create_metric(metric) for metric in run_spec.metrics]) + [
            TokensMetric()
        ]
        stats: List[Stat] = []
        per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = defaultdict(list)
        with htrack_block(f"{len(metrics)} metrics"):
            for metric in metrics:
                with htrack_block(metric):
                    metric_result: MetricResult = metric.evaluate(
                        scenario_state, self.metric_service, self.eval_cache_path
                    )
                    stats.extend(metric_result.aggregated_stats)
                    for key in metric_result.per_instance_stats:
                        per_instance_stats[key].extend(metric_result.per_instance_stats[key])

        # Print out stats
        with htrack_block("Stats"):
            for stat in stats:
                hlog(stat)

        if self.skip_instances:
            hlog("skip_instances was True. Skipping writing results out.")
            return

        # Output benchmarking information and results to files
        write(os.path.join(run_path, "run_spec.json"), json.dumps(asdict(run_spec), indent=2))

        scenario_dict = asdict(scenario)
        scenario_dict["instances"] = [asdict(instance) for instance in scenario_state.instances]
        write_lines(os.path.join(run_path, "scenario.txt"), scenario.render_lines(scenario_state.instances))
        write(os.path.join(run_path, "scenario.json"), json.dumps(scenario_dict, indent=2))

        write_lines(os.path.join(run_path, "scenario_state.txt"), scenario_state.render_lines())
        write(os.path.join(run_path, "scenario_state.json"), json.dumps(asdict(scenario_state), indent=2))

        write_lines(os.path.join(run_path, "metrics.txt"), [str(stat) for stat in stats])
        write(os.path.join(run_path, "metrics.json"), json.dumps([asdict(stat) for stat in stats], indent=2))
        write(
            os.path.join(run_path, "per_instance_metrics.json"),
            json.dumps(
                {str(key): [asdict(stat) for stat in value] for (key, value) in per_instance_stats.items()}, indent=2,
            ),
        )
