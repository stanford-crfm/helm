from abc import ABC
from dataclasses import dataclass, replace
from typing import List, Dict, Tuple, Set, Optional
from math import log, e
from collections import defaultdict

from common.object_spec import ObjectSpec, create_object
from common.general import singleton
from .statistic import Stat, merge_stat
from .augmentations.perturbation_description import PerturbationDescription
from .adapter import (
    AdapterSpec,
    ScenarioState,
    RequestState,
    ADAPT_LANGUAGE_MODELING,
)
from .metric_name import MetricName
from .metric_service import MetricService
from .scenarios.scenario import Instance


@dataclass(unsafe_hash=True)
class PerInstanceStatsKey:
    """
    `PerInstanceStatsKey` is a (instance, trial index) tuple.
    """

    instance: str
    trial_index: int

    def __init__(self, instance: Instance, trial_index: int):
        self.instance = instance.id if instance.id is not None else str(instance)
        self.trial_index = trial_index


@dataclass
class MetricResult:
    """
    `MetricResult` is a wrapper around aggregated statistics (averaged over instances and trial index),
    and per-(instance, trial index) statistics.
    """

    aggregated_stats: List[Stat]

    # Key for per-instance statistics is (instance, trial index), value is list of statistics.
    per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]]


class Metric(ABC):
    """
    A `Metric` takes the results of execution and produces `Stat`s for a
    scenario.

    Note: `Metric` actually right now is a bit of misnomer because it produces many
    `Stat`s, that might be distinct but are computed together.  Eventually we
    might move to a world where there is one (or very few metrics that are domain-independent).
    """

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str
    ) -> MetricResult:
        """
        Main entry point for a `Metric`.  This function groups the single
        list of `RequestState` by training trial and instance, and invokes
        other functions to process those.  This should serve most purposes.

        Any logic that doesn't decompose along instances should go here, such
        as robustness.
        """
        if scenario_state.adapter_spec.method == ADAPT_LANGUAGE_MODELING:
            return self.evaluate_language_modeling(scenario_state, metric_service, eval_cache_path)

        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}  # MetricName -> Stat
        all_per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = {}

        for train_trial_index in range(adapter_spec.num_train_trials):
            trial_stats: Dict[MetricName, Stat] = {}  # Statistics just for this trial
            # Collect statistics per input-metric pair across perturbations
            per_instance_perturbation_stats: Dict[Tuple[MetricName, str], List[Stat]] = defaultdict(list)
            per_metric_instance_ids: Dict[MetricName, Set[str]] = defaultdict(set)  # Collect instance-ids per metric

            # TODO: incorporate disparities (compute difference between average over instances with some tag)
            #       https://github.com/stanford-crfm/benchmarking/issues/48
            for instance_index, instance in enumerate(scenario_state.instances):
                instance_stats = []

                # Evaluate generated request_state
                request_states = scenario_state.get_request_states(train_trial_index, instance, None)
                if len(request_states) != 0:
                    instance_stats.extend(
                        self.evaluate_generation(
                            adapter_spec, singleton(request_states), metric_service, eval_cache_path
                        )
                    )

                # Evaluate the references
                request_states = []
                for reference_index in range(len(instance.references)):
                    request_states.extend(
                        scenario_state.get_request_states(train_trial_index, instance, reference_index)
                    )
                if len(request_states) != 0:
                    instance_stats.extend(
                        self.evaluate_references(adapter_spec, request_states, metric_service, eval_cache_path)
                    )
                all_per_instance_stats[PerInstanceStatsKey(instance, train_trial_index)] = instance_stats

                # Merge these statistics back.
                # TODO: we should add statistics with the individual instances too and serialize them out.
                #       https://github.com/stanford-crfm/benchmarking/issues/49

                for stat in instance_stats:
                    merge_stat(trial_stats, stat)

                    assert instance.id is not None
                    per_metric_instance_ids[stat.name].add(instance.id)
                    # group all perturbations for a specific metric name together
                    if stat.name.perturbation is not None:
                        per_instance_perturbation_stats[(replace(stat.name, perturbation=None), instance.id)].append(
                            stat
                        )

            for (name, instance_id), stats in per_instance_perturbation_stats.items():
                identity_stat: Optional[Stat] = None
                robustness_stat = Stat(
                    replace(name, perturbation=PerturbationDescription(name="robustness", robustness=True))
                )
                fairness_stat = Stat(
                    replace(name, perturbation=PerturbationDescription(name="fairness", fairness=True))
                )
                individual_perturbation_stats: Dict[PerturbationDescription, Stat] = {}

                for stat in stats:  # go through the stats for each perturbation
                    perturbation = stat.name.perturbation
                    assert perturbation is not None
                    if perturbation.name == "identity":
                        assert identity_stat is None  # we should only have one identity stat
                        identity_stat = stat
                    else:
                        if perturbation.robustness:
                            robustness_stat.merge(stat)
                        if perturbation.fairness:
                            fairness_stat.merge(stat)
                        assert perturbation not in individual_perturbation_stats
                        individual_perturbation_stats[perturbation] = Stat(stat.name).merge(stat)  # copy

                for stat in [robustness_stat, fairness_stat, *individual_perturbation_stats.values()]:
                    perturbation = stat.name.perturbation
                    assert perturbation is not None

                    if identity_stat is not None:
                        stat.merge(identity_stat)

                    perturbation = replace(perturbation, name=f"worst_{perturbation.name}")
                    if stat.count > 0:
                        merge_stat(trial_stats, Stat(replace(stat.name, perturbation=perturbation)).add(stat.min))

            for metric_name, instance_ids in per_metric_instance_ids.items():
                merge_stat(trial_stats, Stat(replace(metric_name, name="num_instances")).add(len(instance_ids)))

            # Aggregate the corpus-level metrics
            self.add_perplexity_metrics(trial_stats)

            # We only take the mean value for each trial
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

        # Wrap aggregated and per-instance stats in a MetricResult.
        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate free-form generation.  Override me!"""
        return []

    def evaluate_references(
        self,
        adapter_spec: AdapterSpec,
        reference_request_states: List[RequestState],
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate the references.  Override me!"""
        return []

    def evaluate_language_modeling(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str
    ) -> MetricResult:
        global_stats: Dict[MetricName, Stat] = {}
        # The first and only trial
        trial_stats: Dict[MetricName, Stat] = {}
        # Per-instance stats
        all_per_instance_stats: Dict[PerInstanceStatsKey, List[Stat]] = {}

        for request_state in scenario_state.request_states:
            # Evaluate request_state
            request_stats = self.evaluate_generation(
                scenario_state.adapter_spec, request_state, metric_service, eval_cache_path
            )
            # Use trial index of 0 here since we run only one trial for LM
            all_per_instance_stats[PerInstanceStatsKey(request_state.instance, 0)] = request_stats

            for stat in request_stats:
                merge_stat(trial_stats, stat)

        # Aggregate the corpus-level metrics
        self.add_perplexity_metrics(trial_stats)

        for stat in trial_stats.values():
            merge_stat(global_stats, stat.take_mean())
        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    def add_perplexity_metrics(self, trial_stats: Dict[MetricName, Stat]):
        # TODO: find out the root cause and undo num_X > 0 check
        #       https://github.com/stanford-crfm/benchmarking/issues/350
        for name, stat in list(trial_stats.items()):  # we want to change trial_stats in the loop
            if name.name != "logprob":
                continue
            total_logprob = stat.sum

            num_tokens_name = replace(name, name="num_perplexity_tokens")
            if num_tokens_name in trial_stats:
                num_perplexity_tokens = trial_stats[num_tokens_name].sum
                if num_perplexity_tokens > 0:
                    merge_stat(
                        trial_stats,
                        Stat(replace(name, name="perplexity")).add(e ** (-total_logprob / num_perplexity_tokens)),
                    )

            num_bytes_name = replace(name, name="num_bytes")
            if num_bytes_name in trial_stats:
                num_bytes = trial_stats[num_bytes_name].sum
                if num_bytes > 0:
                    merge_stat(
                        trial_stats, Stat(replace(name, name="bits_per_byte")).add(-total_logprob / num_bytes / log(2))
                    )
                    merge_stat(trial_stats, Stat(replace(name, name="logprob_per_byte")).add(total_logprob / num_bytes))


class MetricSpec(ObjectSpec):
    """Specifies how to create a `Metric`."""

    pass


def create_metric(metric_spec: MetricSpec) -> Metric:
    return create_object(metric_spec)
