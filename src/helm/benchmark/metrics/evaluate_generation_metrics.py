from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
from helm.benchmark.metrics.metric import (
    MetricInterface,
    MetricResult,
    PerInstanceStats,
    add_context,
    compute_worst_case_metrics,
)

from helm.common.general import singleton, parallel_map
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_LANGUAGE_MODELING
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .metric_name import MetricName, MetricContext
from .metric_service import MetricService
from .statistic import Stat, merge_stat


@dataclass(frozen=True)
class Processor:
    """Evaluates an instance."""

    # TODO: not ideal that we have circular dependencies; subclasses of EvaluateGenerationMetric
    # should override the Processor rather than the EvaluateGenerationMetric.
    metric: "EvaluateGenerationMetric"
    metric_service: MetricService
    eval_cache_path: str
    adapter_spec: AdapterSpec

    def process(self, generation_states: List[RequestState]) -> List[Stat]:
        instance_stats: List[Stat] = []

        # Evaluate generated request_state
        if len(generation_states) != 0:
            instance_stats.extend(
                self.metric.evaluate_generation(
                    self.adapter_spec, singleton(generation_states), self.metric_service, self.eval_cache_path
                )
            )

        # Add instance-related context (e.g., split, perturbation) to the metrics
        for i, stat in enumerate(instance_stats):
            instance_stats[i] = add_context(stat, MetricContext.from_instance(generation_states[0].instance))

        return instance_stats


class EvaluateGenerationMetric(MetricInterface, ABC):
    """
    An EvaluateGenerationMetric collects stats from each RequestState independently.

    This class currently assumes that:
    1. There is at most 1 RequestState per Instance per train trial
    2. All the RequestStates we care about have reference_index set to None.

    The stats are then aggregated by:
    1. Within a train trial, compute worst case metrics and take the mean of all metrics.
    2. Report the mean of Stats across train trials.
    """

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        """
        Main entry point for a `Metric`.  This function groups the single
        list of `RequestState` by training trial and instance, and invokes
        other functions to process those.  This should serve most purposes.

        Any logic that doesn't decompose along instances should go here, such
        as robustness.
        """
        assert scenario_state.adapter_spec.method != ADAPT_LANGUAGE_MODELING, (
            "Metric no longer knows how to handle the language modeling adapter. "
            + "All run_specs with that adapter should use LanguageModelingMetric. "
            + "If you are seeing this issue, please file a Github issue."
        )

        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}
        all_per_instance_stats: List[PerInstanceStats] = []

        for train_trial_index in range(adapter_spec.num_train_trials):
            # Construct inputs
            generation_states_sets: List[List[RequestState]] = []
            for instance in scenario_state.instances:
                generation_states_sets.append(scenario_state.get_request_states(train_trial_index, instance, None))

            # Do it!
            processor = Processor(
                metric=self,
                metric_service=metric_service,
                eval_cache_path=eval_cache_path,
                adapter_spec=scenario_state.adapter_spec,
            )
            results: List[List[Stat]] = parallel_map(
                processor.process,
                generation_states_sets,
                parallelism=parallelism,
            )

            # Compute per-instance stats
            per_instance_stats: List[PerInstanceStats] = []
            for instance, stats in zip(scenario_state.instances, results):
                assert instance.id is not None, f"id was none for instance: {instance}"
                # Sometimes a metric (e.g., BiasMetric) doesn't produce any statistics
                if len(stats) > 0:
                    per_instance_stats.append(
                        PerInstanceStats(instance.id, instance.perturbation, train_trial_index, stats)
                    )

            # Aggregate these stats
            trial_stats: Dict[MetricName, Stat] = {}  # Statistics just for this trial
            for instance_stats in results:
                for stat in instance_stats:
                    merge_stat(trial_stats, stat)

            # Compute worst-case metrics.
            # This is here since we want these stats for all metrics and they
            # aggregate across contexts (perturbations).
            worst_case_stats = compute_worst_case_metrics(dict(zip(scenario_state.instances, results)))
            for stat in worst_case_stats:
                merge_stat(trial_stats, stat)

            # We take the mean value for each trial.
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

            all_per_instance_stats.extend(per_instance_stats)

        # Wrap aggregated and per-instance stats in a MetricResult.
        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    @abstractmethod
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate free-form generation."""
        pass
