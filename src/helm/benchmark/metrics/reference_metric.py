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

from helm.common.general import parallel_map
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_LANGUAGE_MODELING
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric_name import MetricName, MetricContext
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat


@dataclass(frozen=True)
class Processor:
    """Evaluates an instance."""

    # TODO: not ideal that we have circular dependencies; subclasses of Metric
    # should override the Processor rather than the Metric.
    metric: "ReferenceMetric"
    metric_service: MetricService
    eval_cache_path: str
    adapter_spec: AdapterSpec

    def process(self, references_states: List[RequestState]) -> List[Stat]:
        instance_stats: List[Stat] = []

        # Evaluate the references
        if len(references_states) == 0:
            return instance_stats
        instance_stats.extend(
            self.metric.evaluate_references(
                self.adapter_spec, references_states, self.metric_service, self.eval_cache_path
            )
        )

        # Add instance-related context (e.g., split, perturbation) to the metrics
        for i, stat in enumerate(instance_stats):
            instance_stats[i] = add_context(stat, MetricContext.from_instance(references_states[0].instance))

        return instance_stats


class ReferenceMetric(MetricInterface, ABC):
    """
    A `Metric` takes the results of execution and produces `Stat`s for a
    scenario.

    Note: `Metric` actually right now is a bit of misnomer because it produces many
    `Stat`s, that might be distinct but are computed together.  Eventually we
    might move to a world where there is one (or very few metrics that are domain-independent).
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
            request_state_sets: List[List[RequestState]] = []
            for instance in scenario_state.instances:
                references_states = []
                for reference_index in range(len(instance.references)):
                    references_states.extend(
                        scenario_state.get_request_states(train_trial_index, instance, reference_index)
                    )
                request_state_sets.append(references_states)

            # Do it!
            processor = Processor(
                metric=self,
                metric_service=metric_service,
                eval_cache_path=eval_cache_path,
                adapter_spec=scenario_state.adapter_spec,
            )
            results: List[List[Stat]] = parallel_map(
                processor.process,
                request_state_sets,
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
    def evaluate_references(
        self,
        adapter_spec: AdapterSpec,
        reference_request_states: List[RequestState],
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate the references.  Override me!"""
        pass
