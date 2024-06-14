import re
from typing import Dict, List

from datasets import load_dataset
import evaluate

from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class UnitxtMetric(MetricInterface):
    ID_PATTERN = re.compile("([a-z]+)([0-9]+)")

    def __init__(self, **kwargs):
        super().__init__()
        dataset_name = ",".join(f"{key}={value}" for key, value in kwargs.items())
        self.dataset = load_dataset("unitxt/data", dataset_name, trust_remote_code=True)

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        # Fetch references from dataset and make them parallel to predictions
        predictions: List[str] = []
        references: List = []
        for request_state in scenario_state.request_states:
            assert request_state.instance.id
            id_match = UnitxtMetric.ID_PATTERN.match(request_state.instance.id)
            assert id_match
            unitxt_split_name = id_match.group(1)
            row_index = int(id_match.group(2))
            references.append(self.dataset[unitxt_split_name][row_index])
            assert request_state.result
            assert len(request_state.result.completions) == 1
            predictions.append(request_state.result.completions[0].text)

        # Compute metrics
        evaluate_results: List[Dict] = evaluate.load("unitxt/metric").compute(
            predictions=predictions, references=references
        )

        # Extract instance metrics
        per_instance_stats: List[PerInstanceStats] = []
        for request_state, evaluate_result in zip(scenario_state.request_states, evaluate_results):
            instance = request_state.instance
            instance_stats: List[Stat] = []
            instance_results = evaluate_result["score"]["instance"]
            for metric_name, metric_score in instance_results.items():
                if metric_name == "score" or metric_name == "score_name":
                    continue
                stat = Stat(
                    MetricName(
                        name=metric_name,
                        split=instance.split,
                        sub_split=instance.sub_split,
                        perturbation=instance.perturbation,
                    )
                )
                if isinstance(metric_score, list):
                    for metric_score_element in metric_score:
                        stat = stat.add(metric_score_element)
                else:
                    stat = stat.add(metric_score)
                instance_stats.append(stat)
            assert instance.id
            per_instance_stats.append(
                PerInstanceStats(
                    instance_id=instance.id,
                    perturbation=instance.perturbation,
                    train_trial_index=request_state.train_trial_index,
                    stats=instance_stats,
                )
            )

        # Extract global metrics
        aggregated_stats: List[Stat] = []
        if len(evaluate_results) > 0:
            global_results = evaluate_results[-1]["score"]["global"]
            for metric_name, metric_score in global_results.items():
                if metric_name == "score" or metric_name == "score_name":
                    continue
                stat = Stat(MetricName(name=metric_name))
                if isinstance(metric_score, list):
                    for metric_score_element in metric_score:
                        stat = stat.add(metric_score_element)
                else:
                    stat = stat.add(metric_score)
                aggregated_stats.append(stat)
        return MetricResult(aggregated_stats=aggregated_stats, per_instance_stats=per_instance_stats)
