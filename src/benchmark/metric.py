from abc import ABC
from dataclasses import dataclass, replace
from typing import List, Dict, Tuple
from math import log, e
from collections import defaultdict

from common.hierarchical_logger import hlog
from common.statistic import Stat, merge_stat
from common.object_spec import ObjectSpec, create_object
from common.general import singleton

from .augmentations.perturbation_description import PerturbationDescription

from .adapter import (
    AdapterSpec,
    ScenarioState,
    RequestState,
    ADAPT_LANGUAGE_MODELING,
    ADAPT_LANGUAGE_MODELING_MINIMAL_PAIRS,
)
from .metric_name import MetricName
from .metric_service import MetricService
from .scenario import Instance, EVAL_SPLITS, TEST_SPLIT


@dataclass
class MetricResult:
    """
    `MetricResult` is a wrapper around aggregated statistics (averaged over instances and trial index),
    and per-(instance, trial index) statistics.
    """

    aggregated_stats: List[Stat]

    # Key for per-instance statistics is (instance, trial index), value is list of statistics.
    # TODO: Consider making the key a dataclass instead.
    per_instance_stats: Dict[Tuple[Instance, int], List[Stat]]


class Metric(ABC):
    """
    A `Metric` takes the results of execution and produces `Stat`s for a
    scenario.

    Note: `Metric` actually right now is a bit of misnomer because it produces many
    `Stat`s, that might be distinct but are computed together.  Eventually we
    might move to a world where there is one (or very few metrics that are domain-independent).
    """

    def evaluate(self, scenario_state: ScenarioState, metric_service: MetricService) -> MetricResult:
        """
        Main entry point for a `Metric`.  This function groups the single
        list of `RequestState` by training trial and instance, and invokes
        other functions to process those.  This should serve most purposes.

        Any logic that doesn't decompose along instances should go here, such
        as robustness.
        """
        if scenario_state.adapter_spec.method == ADAPT_LANGUAGE_MODELING:
            return self.evaluate_language_modeling(scenario_state, metric_service)
        elif scenario_state.adapter_spec.method == ADAPT_LANGUAGE_MODELING_MINIMAL_PAIRS:
            return self.evaluate_language_modeling_minimal_pairs(scenario_state, metric_service)

        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}  # MetricName -> Stat
        all_per_instance_stats: Dict[Tuple[Instance, int], List[Stat]] = {}

        for train_trial_index in range(adapter_spec.num_train_trials):
            trial_stats: Dict[MetricName, Stat] = {}  # Statistics just for this trial
            per_instance_stats: Dict[Tuple[MetricName, str], Stat] = {}  # Statistics for per-instance worst-case metric
            # TODO: incorporate disparities (compute difference between average over instances with some tag)
            #       https://github.com/stanford-crfm/benchmarking/issues/48
            for instance_index, instance in enumerate(scenario_state.instances):
                instance_stats = []
                hlog(f"trial {train_trial_index}: evaluate {instance_index} (total {len(scenario_state.instances)})")

                # Evaluate generated request_state
                request_state = singleton(scenario_state.get_request_states(train_trial_index, instance, None))
                instance_stats.extend(self.evaluate_generation(adapter_spec, request_state, metric_service))

                # Evaluate the references
                request_states = []
                for reference_index in range(len(instance.references)):
                    request_states.extend(
                        scenario_state.get_request_states(train_trial_index, instance, reference_index)
                    )
                instance_stats.extend(self.evaluate_references(adapter_spec, request_states, metric_service))
                all_per_instance_stats[(instance, train_trial_index)] = instance_stats

                # Merge these statistics back.
                # TODO: we should add statistics with the individual instances too and serialize them out.
                #       https://github.com/stanford-crfm/benchmarking/issues/49
                for stat in instance_stats:
                    stat = Stat(replace(stat.name, split=instance.split)).merge(stat)
                    merge_stat(trial_stats, stat)

                    stat = Stat(replace(stat.name, perturbation=PerturbationDescription(name="worst"))).merge(stat)
                    assert instance.id is not None
                    key = (stat.name, instance.id)
                    if key not in per_instance_stats:
                        per_instance_stats[key] = stat
                    else:
                        per_instance_stats[key].merge(stat)

            for (name, instance_id), stat in per_instance_stats.items():
                if stat.count > 0:
                    worst_stat = Stat(stat.name).add(stat.min)
                    merge_stat(trial_stats, worst_stat)

            # Aggregate the corpus-level metrics
            for split in EVAL_SPLITS:
                if (
                    MetricName("logprob", split=split) in trial_stats
                    and MetricName("num_tokens", split=split) in trial_stats
                    and MetricName("num_bytes", split=split) in trial_stats
                ):
                    merge_stat(
                        trial_stats,
                        Stat(MetricName("perplexity", split=split)).add(
                            e
                            ** (
                                -trial_stats[MetricName("logprob", split=split)].sum
                                / trial_stats[MetricName("num_tokens", split=split)].sum
                            )
                        ),
                    )
                    merge_stat(
                        trial_stats,
                        Stat(MetricName("bits_per_byte", split=split)).add(
                            -trial_stats[MetricName("logprob", split=split)].sum
                            / trial_stats[MetricName("num_bytes", split=split)].sum
                            / log(2)
                        ),
                    )
                    merge_stat(
                        trial_stats,
                        Stat(MetricName("logprob_per_byte", split=split)).add(
                            trial_stats[MetricName("logprob", split=split)].sum
                            / trial_stats[MetricName("num_bytes", split=split)].sum
                        ),
                    )

            # We only take the mean value for each trial
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

        # Wrap aggregated and per-instance stats in a MetricResult.
        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Evaluate free-form generation.  Override me!"""
        return []

    def evaluate_references(
        self, adapter_spec: AdapterSpec, reference_request_states: List[RequestState], metric_service: MetricService
    ) -> List[Stat]:
        """Evaluate the references.  Override me!"""
        return []

    def evaluate_language_modeling(self, scenario_state: ScenarioState, metric_service: MetricService) -> MetricResult:
        global_stats: Dict[MetricName, Stat] = {}
        # The first and only trial
        trial_stats: Dict[MetricName, Stat] = {}
        # Per-instance stats
        all_per_instance_stats: Dict[Tuple[Instance, int], List[Stat]] = {}
        # Assume models are only evaluated on the test set
        split: str = TEST_SPLIT

        for request_state in scenario_state.request_states:
            # Evaluate request_state
            request_stats = self.evaluate_generation(scenario_state.adapter_spec, request_state, metric_service)
            # Use trial index of 0 here since we run only one trial for LM
            all_per_instance_stats[(request_state.instance, 0)] = request_stats

            for stat in request_stats:
                stat = Stat(replace(stat.name, split=split)).merge(stat)
                merge_stat(trial_stats, stat)

        # Aggregate the corpus-level metrics
        if (
            MetricName("logprob", split=split) in trial_stats
            and MetricName("num_tokens", split=split) in trial_stats
            and trial_stats[MetricName("num_tokens", split=split)].sum != 0
        ):
            merge_stat(
                trial_stats,
                Stat(MetricName("perplexity", split=split)).add(
                    e
                    ** (
                        -trial_stats[MetricName("logprob", split=split)].sum
                        / trial_stats[MetricName("num_tokens", split=split)].sum
                    )
                ),
            )
            merge_stat(
                trial_stats,
                Stat(MetricName("bits_per_byte", split=split)).add(
                    -trial_stats[MetricName("logprob", split=split)].sum
                    / trial_stats[MetricName("num_bytes", split=split)].sum
                    / log(2)
                ),
            )
            merge_stat(
                trial_stats,
                Stat(MetricName("logprob_per_byte", split=split)).add(
                    trial_stats[MetricName("logprob", split=split)].sum
                    / trial_stats[MetricName("num_bytes", split=split)].sum
                ),
            )

        for stat in trial_stats.values():
            merge_stat(global_stats, stat.take_mean())
        return MetricResult(list(global_stats.values()), all_per_instance_stats)

    def evaluate_language_modeling_minimal_pairs(
        self, scenario_state: ScenarioState, metric_service: MetricService
    ) -> MetricResult:
        """
        This function computes the log probability of both sentences in each minimal pair
        and compares them. If the model assigns a higher log probability to the "good" sentence,
        it is considered correct.

        After evaluating the model on all the minimal pairs in the scenario, the function
        returns an accuracy score.

        This implementation is based on the important assumption that the adaptation process does not
        change the order of the instances and the instance ids are assigned based on the sequential
        order of the instances.
        """
        global_stats: Dict[MetricName, Stat] = {}
        # The first and only trial
        trial_stats: Dict[MetricName, Stat] = {}
        # Per-instance stats
        all_per_instance_stats: Dict[Tuple[Instance, int], List[Stat]] = {}
        # Assume models are only evaluated on the test set
        split: str = TEST_SPLIT

        # The logprobs of good and bad sentences in the dataset
        good_logprobs: defaultdict = defaultdict(float)
        bad_logprobs: defaultdict = defaultdict(float)

        if scenario_state.request_states:
            for request_state in scenario_state.request_states:
                assert request_state.instance.id is not None and request_state.instance.sub_split is not None
                pair_id: int = int(request_state.instance.id.lstrip("id")) // 2
                sub_split: str = request_state.instance.sub_split
                request_stats = self.evaluate_generation(scenario_state.adapter_spec, request_state, metric_service)
                for stat in request_stats:
                    if stat.name == MetricName("logprob"):
                        if sub_split == "good":
                            good_logprobs[pair_id] += stat.sum
                        elif sub_split == "bad":
                            bad_logprobs[pair_id] += stat.sum
                        else:
                            raise Exception(f"Unknown sub_split {sub_split}")
                        continue
                all_per_instance_stats[(request_state.instance, 0)] = request_stats

            accuracy = sum(good_logprobs[pair_id] > bad_logprobs[pair_id] for pair_id in good_logprobs) / len(
                good_logprobs
            )
        else:
            accuracy = 0

        merge_stat(trial_stats, Stat(MetricName("accuracy", split=split)).add(accuracy))

        for stat in trial_stats.values():
            merge_stat(global_stats, stat.take_mean())
        return MetricResult(list(global_stats.values()), all_per_instance_stats)


class MetricSpec(ObjectSpec):
    """Specifies how to create a `Metric`."""

    pass


def create_metric(metric_spec: MetricSpec) -> Metric:
    return create_object(metric_spec)
