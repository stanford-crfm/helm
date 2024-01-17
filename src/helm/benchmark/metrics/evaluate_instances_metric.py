from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, add_context


from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import group_request_states_by_train_trial
from .metric_name import MetricName, MetricContext
from .metric_service import MetricService
from .statistic import Stat, merge_stat


class EvaluateInstancesMetric(MetricInterface, ABC):
    """
    Metric that needs to examine all request states for all instances in the same split with the same perturbations
    in order to determine the Stats.
    """

    def evaluate(
        self, request_states: List[RequestState], metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        """Aggregate over calls to evaluate_instances, which is defined by the subclass.

        1. Each call has all instances for the same train trial, split, and perturbations.
        2. For each train trial, take the mean for each Stat.
        3. Returns Stats built from those means (e.g. the mean in the result is the mean-of-means).
        """
        global_stats: Dict[MetricName, Stat] = {}
        for trial_request_states in group_request_states_by_train_trial(request_states):

            # Aggregate these stats
            trial_stats: Dict[MetricName, Stat] = {}  # Statistics just for this trial

            # Compute statistics that depend on all the `RequestStates` (e.g., bias metrics).
            # Aggregate request states and call evaluate_instances in case the metric needs it.
            grouped_request_states: Dict[MetricContext, List[RequestState]] = defaultdict(list)
            for request_state in trial_request_states:
                grouped_request_states[MetricContext.from_instance(request_state.instance)].append(request_state)
            for context, request_states_for_context in grouped_request_states.items():
                for stat in self.evaluate_instances(request_states_for_context):
                    merge_stat(trial_stats, add_context(stat, context))

            # We take the mean value for each trial.
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

        # Wrap aggregated and per-instance stats in a MetricResult.
        return MetricResult(list(global_stats.values()), [])

    @abstractmethod
    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        """Evaluate all request states directly. Use only if nothing else works."""
        pass
