import os
from typing import Dict, List, Optional

from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.presentation.schema import MetricNameMatcher, RunGroup
from helm.benchmark.presentation.summarize import Run, Summarizer
from helm.benchmark.presentation.table import Cell
from helm.common.codec import from_json


class ToRRRobustnessSummarizer(Summarizer):
    """A Summarizer that computes robustness metrics.

    This Summarizer computes a robustness metrics based on the definition in the ToRR paper.
    The instance-level robustness score for a given model and instance is defined as
    1 - (max_score - min_score) where max_score and min_scores are the maximum and minimum
    scores for the model on that instance across all runs (i.e. across all augmentations
    and serialization formats). The robustness score for a given model and scenario is
    the mean of the model's instance-level robustness score across all instances in that scenario.

    The core HELM framework does not natively support computing metrics that depend on
    per-instance metrics across multiple runs, therefore this special summarizer is needed
    to compute this robustness metic."""

    def __init__(
        self,
        release: Optional[str],
        suites: Optional[List[str]],
        suite: Optional[str],
        schema_path: str,
        output_path: str,
        verbose: bool,
        num_threads: int,
        allow_unknown_models: bool,
    ):
        super().__init__(
            release,
            suites,
            suite,
            schema_path,
            output_path,
            verbose,
            num_threads,
            allow_unknown_models,
        )
        self.run_group_to_model_name_to_robustness: Dict[str, Dict[str, float]] = {}

    PERFORMANCE_METRIC_GROUP_NAME = "performance_metrics"
    ROBUSTNESS_METRIC_GROUP_NAME = "robustness_metrics"
    ROBUSTNESS_METRIC_NAME = "robustness"

    def _get_instance_id_to_performance(
        self, run: Run, performance_metric_matcher: MetricNameMatcher
    ) -> Dict[str, float]:
        with open(os.path.join(run.run_path, "per_instance_stats.json")) as f:
            per_instance_stats = from_json(f.read(), List[PerInstanceStats])
        instance_id_to_performance: Dict[str, float] = {}
        for per_instance_stats_item in per_instance_stats:
            assert per_instance_stats_item.train_trial_index == 0
            assert per_instance_stats_item.perturbation is None
            for stat in per_instance_stats_item.stats:
                if performance_metric_matcher.matches(stat.name):
                    assert per_instance_stats_item.instance_id not in instance_id_to_performance
                    if stat.mean is not None:
                        instance_id_to_performance[per_instance_stats_item.instance_id] = stat.mean

        return instance_id_to_performance

    def _compute_robustness_for_runs(self, runs: List[Run], performance_metric_matcher: MetricNameMatcher) -> float:
        instance_id_to_performances: Dict[str, List[float]] = {}
        for run in runs:
            for instance_id, performance in self._get_instance_id_to_performance(
                run, performance_metric_matcher
            ).items():
                if instance_id not in instance_id_to_performances:
                    instance_id_to_performances[instance_id] = []
                instance_id_to_performances[instance_id].append(performance)
        instance_id_to_robustness: Dict[str, float] = {}
        for instance_id, performances in instance_id_to_performances.items():
            instance_id_to_robustness[instance_id] = 1 - (max(performances) - min(performances))
        return sum(instance_id_to_robustness.values()) / len(instance_id_to_robustness.values())

    def _compute_robustness_for_run_group(self, run_group: RunGroup) -> Dict[str, float]:
        performance_metric_group = self.schema.name_to_metric_group[self.PERFORMANCE_METRIC_GROUP_NAME]
        assert len(performance_metric_group.metrics) == 1
        performance_metric_matcher = performance_metric_group.metrics[0].substitute(run_group.environment)

        group_runs = [run for run in self.runs if run_group.name in run.run_spec.groups]
        model_name_to_runs: Dict[str, List[Run]] = {}

        for run in group_runs:
            model_name = run.run_spec.adapter_spec.model
            if model_name not in model_name_to_runs:
                model_name_to_runs[model_name] = []
            model_name_to_runs[run.run_spec.adapter_spec.model].append(run)

        model_to_robustness: Dict[str, float] = {}
        for model_name, model_runs in model_name_to_runs.items():
            model_to_robustness[model_name] = self._compute_robustness_for_runs(model_runs, performance_metric_matcher)
        return model_to_robustness

    def write_groups(self):
        for run_group in self.schema.run_groups:
            if self.ROBUSTNESS_METRIC_GROUP_NAME and self.PERFORMANCE_METRIC_GROUP_NAME in run_group.metric_groups:
                self.run_group_to_model_name_to_robustness[run_group.name] = self._compute_robustness_for_run_group(
                    run_group
                )
        return super().write_groups()

    def create_cell(
        self,
        runs: List[Run],
        matcher: MetricNameMatcher,
        additional_info: Optional[str],
        hide_value: bool = False,
        is_scenario_table: bool = False,
    ) -> Cell:
        """
        Use the metric name identified by `matcher` to pull out the stats from
        `runs` and return a representation of the average.
        There are four cases:
        1. No matching runs
        2. Matching runs but no matching stats (maybe stat was named incorrectly)
        3. Matching runs, matching stats, but stats have count = 0, so mean is undefined
           (e.g., bias metric ran and computed 0/0)
        4. Matching runs, matching stats, stats with count > 0

        In the first three cases, the cell value is None, but the description distinguishes between these cases.
        """
        if matcher.name != self.ROBUSTNESS_METRIC_NAME:
            return super().create_cell(runs, matcher, additional_info, hide_value, is_scenario_table)

        if len(runs) == 0:
            return Cell(value=None, description="No matching runs")

        # Link the runs that this cell was aggregated from, if this is not a scenario table.
        # Scenario tables link to the runs in the model cells,
        # whereas non-scenario tables link to the runs in the metrics cells.
        run_spec_names: Optional[List] = None
        if not is_scenario_table:
            # Deduplicate run spec names becuase aggregated_run_specs may have duplicated
            # run specs if a run spec belongs to multiple groups.
            run_spec_names = []
            run_spec_names_set = set()
            for run in runs:
                if run.run_spec.name not in run_spec_names_set:
                    run_spec_names.append(run.run_spec.name)
                    run_spec_names_set.add(run.run_spec.name)

        run_group_set = set(runs[0].run_spec.groups) & set(self.run_group_to_model_name_to_robustness.keys())
        assert len(run_group_set) == 1
        run_group = next(iter(run_group_set))

        model_names_set = set(run.run_spec.adapter_spec.model for run in runs)
        assert len(model_names_set) == 1
        model_name = next(iter(model_names_set))

        value = (
            self.run_group_to_model_name_to_robustness[run_group][model_name]
            if self.run_group_to_model_name_to_robustness[run_group]
            and self.run_group_to_model_name_to_robustness[run_group][model_name]
            else None
        )
        description = str(round(value, 3)) if value is not None else ""
        if hide_value:
            value = None
            description = ""
        if additional_info:
            description += "\n" + additional_info
        if self.verbose:
            description += "\n-- ".join(["\nRun specs:", *(run_spec_names or [])])

        return Cell(
            value=value,
            description=description,
            style={},
            run_spec_names=run_spec_names,
        )
