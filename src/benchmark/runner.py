import json
import os
from dataclasses import dataclass, asdict
from typing import List


from common.general import ensure_directory_exists, write, write_lines
from common.hierarchical_logger import hlog, htrack_block
from .adapter_service import AdapterService
from .augmentations.data_augmenter import DataAugmenterSpec
from .metric_service import MetricService
from .scenario import Scenario, ScenarioSpec, create_scenario, Instance
from .adapter import AdapterSpec, Adapter, ScenarioState
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor
from .metric import Metric, MetricSpec, create_metric, Stat
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

    def __init__(self, execution_spec: ExecutionSpec, output_path: str, run_specs: List[RunSpec]):
        self.executor = Executor(execution_spec)
        self.dry_run = execution_spec.dry_run
        self.adapter_service = AdapterService(self.executor.remote_service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.remote_service, execution_spec.auth)
        self.output_path = output_path
        self.run_specs = run_specs
        ensure_directory_exists(self.output_path)

    def run_all(self):
        for run_spec in self.run_specs:
            self.run_one(run_spec)

    def run_one(self, run_spec: RunSpec):
        # Load the scenario
        scenario: Scenario = create_scenario(run_spec.scenario)

        # Decide where to save the raw data (e.g., "output/scenarios/mmlu").
        # This `output_path` will be used when `Adapter` calls `Scenario.get_instances`.
        scenarios_path = os.path.join(self.output_path, "scenarios")
        ensure_directory_exists(scenarios_path)
        scenario.output_path = os.path.join(scenarios_path, scenario.name)
        ensure_directory_exists(scenario.output_path)
        runs_path = os.path.join(self.output_path, "runs", run_spec.name)
        ensure_directory_exists(runs_path)

        # Data preprocessing
        instances: List[Instance] = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(scenario)

        # Adaptation
        adapter = Adapter(run_spec.adapter_spec, self.adapter_service)
        scenario_state: ScenarioState = adapter.adapt(instances)

        # Execution
        scenario_state = self.executor.execute(scenario_state)

        # Apply the metrics
        # When performing a dry run, just estimate the number of tokens instead of calculating the metrics
        metrics: List[Metric] = (
            [TokensMetric()] if self.dry_run else [create_metric(metric) for metric in run_spec.metrics]
        )
        hlog(f"{len(metrics)} metrics")
        stats: List[Stat] = []
        for metric in metrics:
            stats.extend(metric.evaluate(scenario_state, self.metric_service))

        # Print out stats
        with htrack_block("Stats"):
            for stat in stats:
                hlog(stat)

        # Output benchmarking information and results to files
        write(os.path.join(runs_path, "run_spec.json"), json.dumps(asdict(run_spec), indent=2))

        scenario_dict = asdict(scenario)
        scenario_dict["instances"] = [asdict(instance) for instance in scenario_state.instances]
        write_lines(os.path.join(runs_path, "scenario.txt"), scenario.render_lines(scenario_state.instances))
        write(os.path.join(runs_path, "scenario.json"), json.dumps(scenario_dict, indent=2))

        write_lines(os.path.join(runs_path, "scenario_state.txt"), scenario_state.render_lines())
        write(os.path.join(runs_path, "scenario_state.json"), json.dumps(asdict(scenario_state), indent=2))

        write_lines(os.path.join(runs_path, "metrics.txt"), [str(stat) for stat in stats])
        write(os.path.join(runs_path, "metrics.json"), json.dumps([asdict(stat) for stat in stats], indent=2))
