from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

from scenario import Instance
from schemas import Request, RequestResult
from adapter import ScenarioState, RequestState
from statistic import Stat, merge_stat
from object_spec import ObjectSpec, create_object
from general import singleton


class Metric(ABC):
    """
    Evaluates all the results from a `Scenario` and produces numbers.
    """
    def evaluate(self, scenario_state: ScenarioState) -> List[Stat]:
        """Given all the `InstanceResult`s for a `scenario`, compute the metrics and return the list of `MetricResult`s."""
        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[str, Stat] = {}  # name -> Stat

        for train_trial_index in range(adapter_spec.num_train_trials):
            trial_stats: Dict[str, Stat] = {}  # Statistics just for this trial
            # TODO: incorporate robustness (worst case over a group of instances with some tag)
            # TODO: incorporate disparities (compute difference between average over instances with some tag)
            for instance in scenario_state.instances:
                instance_stats = []

                # Evaluate generated request_state
                request_state = singleton(scenario_state.get_request_states(train_trial_index, instance, None))
                instance_stats.extend(self.evaluate_generation(request_state))

                # Ranking
                request_states = [
                    singleton(scenario_state.get_request_states(train_trial_index, instance, reference_index)) \
                    for reference_index in range(len(instance.references))
                ]
                instance_stats.extend(self.evaluate_references(request_states))

                # Merge
                for stat in instance_stats:
                    merge_stat(trial_stats, stat)

            for stat in trial_stats.values():
                merge_stat(global_stats, stat.collapse_uncertainty())

        return global_stats.values()


    def evaluate_generation(self, request_state: RequestState) -> List[Stat]:
        """Evaluate free-form generation.  Override me!"""
        return []


    def evaluate_references(self, reference_request_states: List[RequestState]) -> List[Stat]:
        """Evaluate the references.  Override me!"""
        return []

class MetricSpec(ObjectSpec):
    """Specifies how to create a `Metric`."""
    pass


def create_metric(metric_spec: MetricSpec) -> Metric:
    return create_object(metric_spec)
