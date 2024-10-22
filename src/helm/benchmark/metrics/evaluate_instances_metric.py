from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, add_context


from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric_name import MetricName, MetricContext
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat


class EvaluateInstancesMetric(MetricInterface, ABC):
    """
    Metric that needs to examine all request states for all instances in the same split with the same perturbations
    in order to determine the Stats.
    """

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        """Aggregate over calls to evaluate_instances, which is defined by the subclass.

        1. Each call has all instances for the same train trial, split, and perturbations.
        2. For each train trial, take the mean for each Stat.
        3. Returns Stats built from those means (e.g. the mean in the result is the mean-of-means).
        """
        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}

        for train_trial_index in range(adapter_spec.num_train_trials):

            # Aggregate these stats
            trial_stats: Dict[MetricName, Stat] = {}  # Statistics just for this trial

            # Compute statistics that depend on all the `RequestStates` (e.g., bias metrics).
            # Aggregate request states and call evaluate_instances in case the metric needs it.
            grouped_request_states: Dict[MetricContext, List[RequestState]] = defaultdict(list)
            for instance in scenario_state.instances:
                # TODO: do we need to support reference_index that is not None?
                grouped_request_states[MetricContext.from_instance(instance)].extend(
                    scenario_state.get_request_states(train_trial_index, instance, None)
                )
            for context, request_states in grouped_request_states.items():
                for stat in self.evaluate_instances(request_states, eval_cache_path):
                    merge_stat(trial_stats, add_context(stat, context))

            # We take the mean value for each trial.
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

        # Wrap aggregated and per-instance stats in a MetricResult.
        return MetricResult(list(global_stats.values()), [])

    @abstractmethod
    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        """Evaluate all request states directly. Use only if nothing else works."""
        pass
