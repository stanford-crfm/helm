import json
import os
import typing
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, cast

from helm.benchmark.augmentations.perturbation import PerturbationDescription
from helm.benchmark.metrics.metric_name import MetricName
from helm.common.general import ensure_directory_exists, write, asdict_without_nones
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.cache import cache_stats
from .augmentations.data_augmenter import DataAugmenterSpec
from .scenarios.scenario import Scenario, ScenarioSpec, create_scenario, Instance, with_instance_ids
from .adapter import AdapterSpec, Adapter, RequestState, ScenarioState, slimmed_scenario_state
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor
from .metrics.metric_service import MetricService
from .metrics.metric import Metric, MetricSpec, MetricResult, PerInstanceStats, create_metric, Stat
from .metrics.tokens_metric import TokensMetric
from .window_services.tokenizer_service import TokenizerService
from helm.benchmark.presentation.schema import read_schema


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


@dataclass(frozen=True)
class SlimPrediction:
    """
    Captures a unit of evaluation for displaying in the web frontend.
    """

    # (instance_id, perturbation, train_trial_index) is a unique key for this prediction.
    instance_id: str
    """ID of the Instance"""

    perturbation: Optional[PerturbationDescription]
    """Description of the Perturbation that was applied"""

    train_trial_index: int
    """Which replication"""

    predicted_text: str
    """Prediction text"""

    truncated_predicted_text: Optional[str]
    """The truncated prediction text, if truncation is required by the Adapter method."""

    mapped_output: Optional[str]
    """The mapped output, if an output mapping exists and the prediction can be mapped"""

    reference_index: Optional[int]
    """Which reference of the instance we're evaluating (if any)"""

    stats: Dict[str, float]
    """Statistics computed from the predicted output"""


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

        run_path: str = os.path.join(self.runs_path, run_spec.name)
        ensure_directory_exists(run_path)

        adapter = Adapter(run_spec.adapter_spec, self.tokenizer_service)

        instances: List[Instance]
        if not self.skip_instances:
            # Create the instances of the scenario
            with htrack_block("scenario.get_instances"):
                instances = scenario.get_instances()

            # Give each instance a unique ID
            instances = with_instance_ids(instances)

            # Sample only as many as we need
            instances = adapter.sample_instances(instances)

            # Data preprocessing
            instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
                instances, self.executor.execution_spec.parallelism
            )
        else:
            instances = []

        # Adapt (convert to requests)
        scenario_state: ScenarioState = adapter.adapt(instances, self.executor.execution_spec.parallelism)

        # Execute (fill up results)
        scenario_state = self.executor.execute(scenario_state)

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

        # Print out the number of stats
        hlog(f"Generated {len(stats)} stats.")

        if self.skip_instances:
            hlog("skip_instances was True. Skipping writing results out.")
            return

        # Output benchmarking information and results to files
        write(os.path.join(run_path, "run_spec.json"), json.dumps(asdict_without_nones(run_spec), indent=2))

        # Write out scenario
        write(os.path.join(run_path, "scenario.json"), json.dumps(asdict_without_nones(scenario), indent=2))

        # Write scenario state (including a slim version that is much smaller)
        write(os.path.join(run_path, "scenario_state.json"), json.dumps(asdict_without_nones(scenario_state), indent=2))
        write(
            os.path.join(run_path, "scenario_state_slim.json"),
            json.dumps(asdict_without_nones(slimmed_scenario_state(scenario_state)), indent=2),
        )

        write(
            os.path.join(run_path, "stats.json"), json.dumps([asdict_without_nones(stat) for stat in stats], indent=2)
        )
        write(
            os.path.join(run_path, "per_instance_stats.json"),
            json.dumps(list(map(asdict_without_nones, per_instance_stats)), indent=2),
        )

        write(
            os.path.join(run_path, "instances.json"),
            json.dumps([asdict_without_nones(instance) for instance in scenario_state.instances], indent=2),
        )

        schema = read_schema()
        metric_groups_by_name = {metric_group.name: metric_group for metric_group in schema.metric_groups}
        run_groups_by_name = {run_group.name: run_group for run_group in schema.run_groups}

        metric_names_from_schema: Set[str] = set(
            metric_name_matcher.substitute(run_groups_by_name[run_group_name].environment).name
            for run_group_name in run_spec.groups
            for metric_group_name in run_groups_by_name[run_group_name].metric_groups
            for metric_name_matcher in metric_groups_by_name[metric_group_name].metrics
            if not metric_name_matcher.perturbation_name
        )
        if run_spec.adapter_spec.method.startswith("multiple_choice_separate_"):
            metric_names_from_schema.add("predicted_index")

        stats_by_trial: Dict[Tuple[str, Optional[PerturbationDescription], int], Dict[str, float]] = defaultdict(dict)
        for original_stats in per_instance_stats:
            stats_dict: Dict[str, float] = {
                original_stat.name.name: cast(float, original_stat.mean)
                for original_stat in original_stats.stats
                if original_stat.name.name in metric_names_from_schema
            }
            key = (original_stats.instance_id, original_stats.perturbation, original_stats.train_trial_index)
            stats_by_trial[key].update(stats_dict)

        predictions = []
        for request_state in scenario_state.request_states:
            assert request_state.instance.id is not None
            assert request_state.result is not None

            predicted_text = (
                request_state.result.completions[0].text
                if request_state.result is not None or request_state.result.completions
                else None
            )
            mapped_output = (
                request_state.output_mapping.get(predicted_text.strip()) if request_state.output_mapping else None
            )

            stats_key = (
                request_state.instance.id,
                request_state.instance.perturbation,
                request_state.train_trial_index,
            )
            predictions.append(
                SlimPrediction(
                    instance_id=request_state.instance.id,
                    perturbation=request_state.instance.perturbation,
                    train_trial_index=request_state.train_trial_index,
                    predicted_text=predicted_text,
                    truncated_predicted_text=self._truncate_predicted_text(
                        predicted_text, request_state, run_spec.adapter_spec
                    ),
                    mapped_output=mapped_output,
                    reference_index=request_state.reference_index,
                    stats=stats_by_trial[stats_key],
                )
            )
        write(
            os.path.join(run_path, "predictions.json"),
            json.dumps(list(map(asdict_without_nones, predictions)), indent=2),
        )
        cache_stats.print_status()

    def _truncate_predicted_text(
        self, predicted_text: str, request_state: RequestState, adapter_spec: AdapterSpec
    ) -> Optional[str]:
        method = adapter_spec.method
        prefix = ""
        if method.startswith("multiple_choice_separate_"):
            prefix = request_state.instance.input
        elif method == "language_modeling":
            if request_state.result is not None and request_state.result.completions:
                tokens = request_state.result.completions[0].tokens
                if tokens:
                    first_token = tokens[0]
                    if not first_token.top_logprobs:
                        prefix = first_token.text
                    else:
                        hlog("WARNING: top_logprobs was not empty, skipping predicted_text truncation")
        if prefix:
            predicted_text = predicted_text.strip()
            prefix = prefix.strip()
            if predicted_text.startswith(prefix):
                return predicted_text[len(prefix) :].strip()
        return None
