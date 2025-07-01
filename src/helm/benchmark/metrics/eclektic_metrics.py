from collections import defaultdict
from typing import List, Dict

import pandas as pd

from helm.common.object_spec import ObjectSpec, create_object
from helm.common.general import singleton, parallel_map

from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_LANGUAGE_MODELING
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.metrics.metric_name import MetricName, MetricContext
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import (
    Metric,
    MetricResult,
    PerInstanceStats,
    Processor,
    add_context,
    compute_worst_case_metrics,
)


class EclekticMetric(Metric):
    """Score metrics for Eclektic."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:

        assert request_state.annotations
        scores = request_state.annotations["eclektic_autograder"]

        return [Stat(MetricName("accuracy")).add(scores["correct"])]

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

        data_rows: List[Dict[str, object]] = []

        for train_trial_index in range(adapter_spec.num_train_trials):
            # Construct inputs
            generation_state_sets: List[List[RequestState]] = []
            for instance in scenario_state.instances:
                generation_state_sets.append(scenario_state.get_request_states(train_trial_index, instance, None))

            # Do it!
            processor = Processor(
                metric=self,
                metric_service=metric_service,
                eval_cache_path=eval_cache_path,
                adapter_spec=scenario_state.adapter_spec,
            )
            results: List[List[Stat]] = parallel_map(
                processor.process,
                generation_state_sets,
                parallelism=parallelism,
            )
            # ----------------------------------------------------------
            # Record the fields we need for the corpus‑level calculations
            # ----------------------------------------------------------
            for instance, req_states in zip(scenario_state.instances, generation_state_sets):
                if not req_states:
                    continue  # Defensive guard
                rs = req_states[0]  # Exactly one RequestState per instance
                ann = rs.annotations.get("eclektic_autograder", {})

                data_rows.append(
                    {
                        "q_id": instance.extra_data.get("q_id"),
                        "lang": instance.extra_data.get("lang"),
                        "original_lang": instance.extra_data.get("original_lang"),
                        "correct": bool(ann.get("correct", False)),
                    }
                )

            # ----------------------------------------------------------
            # (Mostly boilerplate) accumulate per‑instance and trial stats
            # ----------------------------------------------------------
            per_instance_stats: List[PerInstanceStats] = []
            for instance, stats in zip(scenario_state.instances, results):
                if stats:
                    per_instance_stats.append(
                        PerInstanceStats(instance.id, instance.perturbation, train_trial_index, stats)
                    )

            trial_stats: Dict[MetricName, Stat] = {}
            for instance_stats in results:
                for stat in instance_stats:
                    merge_stat(trial_stats, stat)

            # Derivations grouped by context (unchanged pattern)
            grouped_trial_stats: Dict[MetricContext, Dict[MetricName, Stat]] = defaultdict(dict)
            for metric_name, stat in trial_stats.items():
                grouped_trial_stats[MetricContext.from_metric_name(metric_name)][metric_name] = stat
            for context, stats_dict in grouped_trial_stats.items():
                for stat in self.derive_stats(stats_dict):
                    merge_stat(trial_stats, add_context(stat, context))

            grouped_per_instance_stats: Dict[MetricContext, Dict[Instance, List[Stat]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for instance, stats in zip(scenario_state.instances, results):
                for stat in stats:
                    grouped_per_instance_stats[MetricContext.from_instance(instance)][instance].append(stat)
            for context, instance_dict in grouped_per_instance_stats.items():
                for stat in self.derive_per_instance_stats(instance_dict):
                    merge_stat(trial_stats, add_context(stat, context))

            worst_case_stats = compute_worst_case_metrics(dict(zip(scenario_state.instances, results)))
            for stat in worst_case_stats:
                merge_stat(trial_stats, stat)

            # Fold this trial's mean stats into the global aggregation
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

            all_per_instance_stats.extend(per_instance_stats)

        # --------------------------------------------------------------
        # Compute corpus‑level *overall* and *overall_transfer*
        # --------------------------------------------------------------
        if data_rows:  # Skip if evaluation somehow produced no data
            data = pd.DataFrame(data_rows)

            # Questions answered correctly in their *original* language
            correct_in_lang_qids = set(
                data[(data["correct"]) & (data["lang"] == data["original_lang"])]["q_id"].tolist()
            )

            # ------------------ overall (translated only) ------------------
            scored_data = data[data["lang"] != data["original_lang"]]
            if not scored_data.empty:
                overall_successes = scored_data[
                    (scored_data["correct"]) & (scored_data["q_id"].isin(correct_in_lang_qids))
                ]
                overall_score = len(overall_successes) / len(scored_data)
            else:
                overall_score = 0.0
            merge_stat(global_stats, Stat(MetricName("overall")).add(overall_score))

            # ------------- overall_transfer (all languages) ---------------
            transfer_data = data[data["q_id"].isin(correct_in_lang_qids)]
            if not transfer_data.empty:
                transfer_successes = transfer_data[
                    (transfer_data["correct"]) & (transfer_data["q_id"].isin(correct_in_lang_qids))
                ]
                transfer_score = len(transfer_successes) / len(transfer_data)
            else:
                transfer_score = 0.0
            merge_stat(
                global_stats,
                Stat(MetricName("overall_transfer")).add(transfer_score),
            )

        return MetricResult(list(global_stats.values()), all_per_instance_stats)
