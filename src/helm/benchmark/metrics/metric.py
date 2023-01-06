from abc import ABC
from dataclasses import dataclass, replace
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Iterable, Set

from helm.common.object_spec import ObjectSpec, create_object
from helm.common.general import singleton, parallel_map
from helm.benchmark.augmentations.perturbation_description import (
    PerturbationDescription,
    PERTURBATION_ORIGINAL,
    PERTURBATION_WORST,
)
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_LANGUAGE_MODELING
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.scenarios.scenario import Instance
from .metric_name import MetricName, MetricContext
from .metric_service import MetricService
from .statistic import Stat, merge_stat


@dataclass(frozen=True)
class PerInstanceStats:
    """
    Captures a unit of evaluation.
    """

    # Uniquely identifies the input instance
    instance_id: str
    perturbation: Optional[PerturbationDescription]
    train_trial_index: int
    """Which replication"""

    stats: List[Stat]
    """Statistics computed from the predicted output"""


@dataclass
class MetricResult:
    """
    `MetricResult` is a wrapper around aggregated statistics (averaged over instances and trial index),
    and per-(instance, trial index) statistics.
    """

    aggregated_stats: List[Stat]
    per_instance_stats: List[PerInstanceStats]


@dataclass(frozen=True)
class RequestStateSet:
    """All the request states relevant to a given instance"""

    instance: Instance
    generation_states: List[RequestState]
    references_states: List[RequestState]


@dataclass(frozen=True)
class Processor:
    """Evaluates an instance."""

    # TODO: not ideal that we have circular dependencies; subclasses of Metric
    # should override the Processor rather than the Metric.
    metric: "Metric"
    metric_service: MetricService
    eval_cache_path: str
    adapter_spec: AdapterSpec

    def process(self, request_state_set: RequestStateSet) -> List[Stat]:
        instance_stats: List[Stat] = []

        # Evaluate generated request_state
        generation_states = request_state_set.generation_states
        if len(generation_states) != 0:
            instance_stats.extend(
                self.metric.evaluate_generation(
                    self.adapter_spec, singleton(generation_states), self.metric_service, self.eval_cache_path
                )
            )

        # Evaluate the references
        references_states = request_state_set.references_states
        if len(references_states) != 0:
            instance_stats.extend(
                self.metric.evaluate_references(
                    self.adapter_spec, references_states, self.metric_service, self.eval_cache_path
                )
            )

        # Add instance-related context (e.g., split, perturbation) to the metrics
        for i, stat in enumerate(instance_stats):
            instance_stats[i] = add_context(stat, MetricContext.from_instance(request_state_set.instance))

        return instance_stats


class Metric(ABC):
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
        if scenario_state.adapter_spec.method == ADAPT_LANGUAGE_MODELING:
            return self.evaluate_language_modeling(scenario_state, metric_service, eval_cache_path)

        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}  # MetricName -> Stat
        all_per_instance_stats: List[PerInstanceStats] = []

        for train_trial_index in range(adapter_spec.num_train_trials):
            # Construct inputs
            request_state_sets: List[RequestStateSet] = []
            for instance in scenario_state.instances:
                generation_states = scenario_state.get_request_states(train_trial_index, instance, None)
                references_states = []
                for reference_index in range(len(instance.references)):
                    references_states.extend(
                        scenario_state.get_request_states(train_trial_index, instance, reference_index)
                    )
                request_state_set = RequestStateSet(
                    instance=instance, generation_states=generation_states, references_states=references_states
                )
                request_state_sets.append(request_state_set)

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

            # Derive new statistics based on existing stats by calling `derive_stats` (e.g., for perplexity).
            # Group stats according to the context (e.g., split, perturbation),
            # i.e., non-name part of the MetricName, and call `derive_stats` on
            # each grouping.
            grouped_trial_stats: Dict[MetricContext, Dict[MetricName, Stat]] = defaultdict(dict)
            for metric_name, stat in trial_stats.items():
                grouped_trial_stats[MetricContext.from_metric_name(metric_name)][metric_name] = stat  # group by context
            for context, stats_dict in grouped_trial_stats.items():
                for stat in self.derive_stats(stats_dict):
                    # we could potentially allow derive_stats to overwrite context, but this feels more robust
                    merge_stat(trial_stats, add_context(stat, context))  # add correct context

            # Derive new per-instance statistics by calling `derive_per_instance_stats` (e.g., for calibration).
            # Again, group stats according to the context before calling
            # `derive_per_instance_stats`.
            grouped_per_instance_stats: Dict[MetricContext, Dict[Instance, List[Stat]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for instance, stats in zip(scenario_state.instances, results):
                for stat in stats:
                    grouped_per_instance_stats[MetricContext.from_instance(instance)][instance].append(stat)
            for context, instance_dict in grouped_per_instance_stats.items():
                # Here, we assume that derive_per_instance_stats only computes trial_stats-level metrics
                # (instance-level metrics should be computed in the evaluate_{generation,references} anyway).
                for stat in self.derive_per_instance_stats(instance_dict):
                    merge_stat(trial_stats, add_context(stat, context))

            # Compute statistics that depend on all the `RequestStates` (e.g., bias metrics).
            # Aggregate request states and call evaluate_instances in case the metric needs it.
            grouped_request_states: Dict[MetricContext, List[RequestState]] = defaultdict(list)
            for instance in scenario_state.instances:
                # TODO: do we need to support reference_index that is not None?
                grouped_request_states[MetricContext.from_instance(instance)].extend(
                    scenario_state.get_request_states(train_trial_index, instance, None)
                )
            for context, request_states in grouped_request_states.items():
                for stat in self.evaluate_instances(request_states):
                    merge_stat(trial_stats, add_context(stat, context))

            # Compute worst-case metrics.
            # This is here since we want these stats for all metrics and they
            # aggregate across contexts (perturbations).
            worst_case_stats = self.compute_worst_case_metrics(dict(zip(scenario_state.instances, results)))
            for stat in worst_case_stats:
                merge_stat(trial_stats, stat)

            # We take the mean value for each trial.
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

            all_per_instance_stats.extend(per_instance_stats)

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

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        """Evaluate all request states directly. Use only if nothing else works.  Override me!"""
        return []

    def derive_stats(self, stats_dict: Dict[MetricName, Stat]) -> List[Stat]:
        """Derive stats based on existing stats, e.g., for perplexity. Override me!"""
        return []

    def derive_per_instance_stats(self, per_instance_stats: Dict[Instance, List[Stat]]) -> List[Stat]:
        """Derive stats based on existing per-instance stats, e.g., for calibration. Override me!"""
        return []

    def evaluate_language_modeling(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str
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

    def compute_worst_case_metrics(self, per_instance_stats: Dict[Instance, List[Stat]]) -> List[Stat]:
        """
        For each instance, we compute the worst case perfomance between each perturbation and the non-perturbed input
        (perturbation=None). This allows us to reason about the invariances of a model as opposed to just looking
        at its performance on perturbed inputs. We also compute the worst case performance across all robustness-related
        and fairness-related perturbations (including the original input in both).

        For each such worst-case metric, we record a `before_` metric that aggregates the performance on the
        non-perturbed version of the corresponding inputs.

        We return the aggregate metrics across instances. Note that none of these metrics make a lot of sense if the
        original, un-perturbed version of an Instance is not included in a scenario (i.e., we want
        `include_original=True`).
        """
        # Collect statistics per input-metric pair across perturbations
        per_instance_perturbation_stats: Dict[Tuple[MetricName, str], List[Stat]] = defaultdict(list)
        for instance, stats in per_instance_stats.items():
            for stat in stats:
                assert instance.id is not None
                # Group all perturbations for a specific metric name together
                per_instance_perturbation_stats[(replace(stat.name, perturbation=None), instance.id)].append(stat)

        # Compute worst perturbation stats
        derived_stats_dict: Dict[MetricName, Stat] = {}
        for (metric_name, instance_id), stats in per_instance_perturbation_stats.items():
            original_stat: Optional[Stat] = None
            robustness_stat = Stat(
                replace(metric_name, perturbation=PerturbationDescription(name="robustness", robustness=True))
            )
            fairness_stat = Stat(
                replace(metric_name, perturbation=PerturbationDescription(name="fairness", fairness=True))
            )
            individual_perturbation_stats: Dict[PerturbationDescription, Stat] = {}

            for stat in stats:  # go through all the perturbations of the instance and merge relevant stats
                perturbation = stat.name.perturbation
                if perturbation is None:
                    assert original_stat is None  # we should only have one original stat
                    original_stat = stat
                else:
                    if perturbation.robustness:
                        robustness_stat.merge(stat)
                    if perturbation.fairness:
                        fairness_stat.merge(stat)
                    assert perturbation not in individual_perturbation_stats, perturbation
                    individual_perturbation_stats[perturbation] = Stat(stat.name).merge(stat)  # copy

            for stat in [robustness_stat, fairness_stat, *individual_perturbation_stats.values()]:
                perturbation = stat.name.perturbation
                assert perturbation is not None

                if original_stat is not None:
                    stat.merge(original_stat)
                    if perturbation.name not in ["robustness", "fairness"]:
                        before = replace(perturbation, computed_on=PERTURBATION_ORIGINAL)
                        merge_stat(
                            derived_stats_dict, Stat(replace(stat.name, perturbation=before)).merge(original_stat)
                        )

                # keep the minimum performance for each input
                worst = replace(perturbation, computed_on=PERTURBATION_WORST)
                if stat.count > 0:
                    # TODO: take stat.max if lower_is_better = True
                    merge_stat(derived_stats_dict, Stat(replace(stat.name, perturbation=worst)).add(stat.min))
        return list(derived_stats_dict.values())


class MetricSpec(ObjectSpec):
    """Specifies how to create a `Metric`."""

    pass


def create_metric(metric_spec: MetricSpec) -> Metric:
    return create_object(metric_spec)


def get_all_stats_by_name(stats: Iterable[Stat], name: str) -> List[Stat]:
    """Returns a list of all stats with the specified name."""

    def matches(stat):
        return stat.name.name == name

    return list(filter(matches, stats))


def get_unique_stat_by_name(stats: Iterable[Stat], name: str) -> Optional[Stat]:
    """Returns the unique stat with the specified name or None if it's not there."""
    matching_stats: List[Stat] = get_all_stats_by_name(stats, name)
    if len(matching_stats) == 0:
        return None
    return singleton(matching_stats)


def add_context(stat: Stat, context: MetricContext) -> Stat:
    """Populate the fields of the Stat with the context info (e.g., split, perturbation) from the instance."""
    return Stat(
        replace(stat.name, split=context.split, sub_split=context.sub_split, perturbation=context.perturbation)
    ).merge(stat)
