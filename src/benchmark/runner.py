import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from common.general import ensure_directory_exists, write, write_lines, asdict_without_nones
from common.hierarchical_logger import hlog, htrack_block
from .augmentations.data_augmenter import DataAugmenterSpec
from .scenarios.scenario import Scenario, ScenarioSpec, create_scenario, Instance
from .adapter import AdapterSpec, Adapter, ScenarioState
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor
from .metrics.metric_service import MetricService
from .metrics.metric import Metric, MetricSpec, MetricResult, PerInstanceStatsKey, create_metric, Stat
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
        run_specs: List[RunSpec],
        skip_instances: bool,
    ):
        self.executor = Executor(execution_spec)
        self.dry_run: bool = execution_spec.dry_run
        self.tokenizer_service = TokenizerService(self.executor.service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.service, execution_spec.auth)
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
        scenario: Scenario = create_scenario(run_spec.scenario_spec)

        # This `output_path` will be used when `Adapter` calls `Scenario.get_instances`.
        scenario.output_path = os.path.join(self.scenarios_path, scenario.name)
        ensure_directory_exists(scenario.output_path)
        scenario.definition_path = scenario.get_definition_path()
        run_path: str = os.path.join(self.runs_path, run_spec.name)
        ensure_directory_exists(run_path)

        # Data preprocessing
        if not self.skip_instances:
            instances: List[Instance] = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
                scenario, self.executor.execution_spec.parallelism
            )
        else:
            instances = []

        # Adaptation
        adapter = Adapter(run_spec.adapter_spec, self.tokenizer_service)
        scenario_state: ScenarioState = adapter.adapt(instances)

        # Execution
        scenario_state = self.executor.execute(scenario_state)

        # Apply the metrics
        # When performing a dry run, only estimate the number of tokens instead
        # of calculating the metrics.
        metrics: List[Metric] = (
            [] if self.dry_run else [create_metric(metric_spec) for metric_spec in run_spec.metric_specs]
        ) + [TokensMetric()]
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
        hlog(f"Generated {len(stats)} stats")

        if self.skip_instances:
            hlog("skip_instances was True. Skipping writing results out.")
            return

        # Output benchmarking information and results to files
        write(os.path.join(run_path, "run_spec.json"), json.dumps(asdict_without_nones(run_spec), indent=2))

        scenario_dict = asdict_without_nones(scenario)
        scenario_dict["instances"] = [asdict_without_nones(instance) for instance in scenario_state.instances]
        write_lines(os.path.join(run_path, "scenario.txt"), scenario.render_lines(scenario_state.instances))
        write(os.path.join(run_path, "scenario.json"), json.dumps(scenario_dict, indent=2))

        write_lines(os.path.join(run_path, "scenario_state.txt"), scenario_state.render_lines())
        write(os.path.join(run_path, "scenario_state.json"), json.dumps(asdict_without_nones(scenario_state), indent=2))

        write_lines(os.path.join(run_path, "stats.txt"), [str(stat) for stat in stats])
        write(
            os.path.join(run_path, "stats.json"), json.dumps([asdict_without_nones(stat) for stat in stats], indent=2)
        )
        write(
            os.path.join(run_path, "per_instance_stats.json"),
            json.dumps(
                {
                    str(key): [asdict_without_nones(stat) for stat in value]
                    for (key, value) in per_instance_stats.items()
                },
                indent=2,
            ),
        )
