from collections import defaultdict
from typing import List, Dict, Set

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.basic_metrics import (
    compute_language_modeling_metrics,
    compute_perplexity_metrics,
    compute_request_state_metrics,
)
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats, add_context
from helm.benchmark.metrics.metric_name import MetricContext, MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat


class LanguageModelingMetric(MetricInterface):
    """
    Defines the basic metrics available when using the ADAPT_LANGUAGE_MODELING adapter.
    This is parallel to BasicMetric and produces many of the same Stats.
    """

    def __init__(self, names: List[str]):
        self.names: List[str] = names
        self.efficiency_metric = EfficiencyMetric()

    def __repr__(self):
        return "LanguageModelingMetric"

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        global_stats: Dict[MetricName, Stat] = {}
        # The first and only trial
        trial_stats: Dict[MetricName, Stat] = {}
        # Per-instance stats
        all_per_instance_stats: List[PerInstanceStats] = []
        instance_ids_per_context: Dict[MetricContext, Set[str]] = defaultdict(set)

        for request_state in scenario_state.request_states:
            # Evaluate request_state
            request_stats = self.evaluate_generation(
                scenario_state.adapter_spec, request_state, metric_service, eval_cache_path
            )

            # Add instance-related context (e.g., split, perturbation) to the metrics
            for i, stat in enumerate(request_stats):
                context = MetricContext.from_instance(request_state.instance)
                request_stats[i] = add_context(stat, context)
                assert request_state.instance.id is not None
                instance_ids_per_context[context].add(request_state.instance.id)

            # Use trial index of 0 here since we run only one trial for LM
            assert request_state.instance.id is not None
            all_per_instance_stats.append(
                PerInstanceStats(request_state.instance.id, request_state.instance.perturbation, 0, request_stats)
            )

            for stat in request_stats:
                merge_stat(trial_stats, stat)

        # group stats according to the context (e.g., split, perturbation) and call derive_stats on each grouping
        grouped_trial_stats: Dict[MetricContext, Dict[MetricName, Stat]] = defaultdict(dict)
        for metric_name, stat in trial_stats.items():
            grouped_trial_stats[MetricContext.from_metric_name(metric_name)][metric_name] = stat  # group by context

        for context, stats_dict in grouped_trial_stats.items():
            for stat in self.derive_stats(stats_dict):
                merge_stat(trial_stats, add_context(stat, context))
            # keep track of how many instances are in each subset
            num_instances_stat = Stat(MetricName("num_instances")).add(len(instance_ids_per_context[context]))
            merge_stat(trial_stats, add_context(num_instances_stat, context))

        for stat in trial_stats.values():
            merge_stat(global_stats, stat.take_mean())
        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Compute all metrics."""
        stats: List[Stat] = []
        stats.extend(compute_request_state_metrics(self.efficiency_metric, adapter_spec, request_state, metric_service))
        stats.extend(compute_language_modeling_metrics(adapter_spec, request_state, metric_service))

        return stats

    def derive_stats(self, stats_dict: Dict[MetricName, Stat]) -> List[Stat]:
        """Derive perplexity metrics if applicable. We don't worry about splits and perturbations here."""
        derived_stats: List[Stat] = []
        derived_stats.extend(compute_perplexity_metrics(stats_dict))
        return derived_stats
