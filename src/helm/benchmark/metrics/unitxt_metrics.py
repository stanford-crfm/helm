import numbers
import re
from typing import Any, Dict, List, Optional, Set, Union
from threading import Lock

from datasets import Dataset, IterableDataset

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import MetricInterface, MetricMetadata, MetricResult, PerInstanceStats
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hwarn
from helm.common.optional_dependencies import handle_module_not_found_error

try:
    from unitxt import evaluate, load_dataset
    from unitxt.api import load_recipe
    from unitxt.artifact import fetch_artifact
    from unitxt.base_metric import Metric

except ModuleNotFoundError as e:
    handle_module_not_found_error(e, suggestions=["unitxt"])


class UnitxtMetric(MetricInterface):
    ID_PATTERN = re.compile("([a-z]+)([0-9]+)")

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._lock = Lock()
        self._dataset_dict: Optional[Dict[str, Union[Dataset, IterableDataset]]] = None

    def get_dataset_dict(self) -> Dict[str, Union[Dataset, IterableDataset]]:
        if not self._dataset_dict:
            with self._lock:
                if len(self.kwargs) == 1 and "recipe" in self.kwargs:
                    unitxt_dataset = load_dataset(self.kwargs["recipe"])
                else:
                    unitxt_dataset = load_dataset(**self.kwargs)

                if isinstance(unitxt_dataset, dict):
                    self._dataset_dict = unitxt_dataset
                else:
                    if "split" in self.kwargs:
                        self._dataset_dict = {self.kwargs["split"]: unitxt_dataset}
                    else:
                        raise Exception(
                            "Expected Unitxt `load_dataset()` to return a dict because `split` was not specified"
                        )

        return self._dataset_dict

    def get_split_to_request_states(self, scenario_state: ScenarioState) -> Dict[str, List[RequestState]]:
        split_to_request_states: Dict[str, List[RequestState]] = {}
        for request_state in scenario_state.request_states:
            split = request_state.instance.split
            assert split is not None
            if split not in split_to_request_states:
                split_to_request_states[split] = []
            split_to_request_states[split].append(request_state)
        return split_to_request_states

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        per_instance_stats: List[PerInstanceStats] = []
        aggregated_stats: List[Stat] = []
        non_number_instance_metric_names: Set[str] = set()

        # Fetch references from dataset and make them parallel to predictions
        dataset_dict = self.get_dataset_dict()
        split_to_request_states = self.get_split_to_request_states(scenario_state)
        for helm_split, request_states in split_to_request_states.items():
            # Fetch references from dataset and make them parallel to predictions
            predictions: List[str] = []
            references: List[Any] = []
            for request_state in request_states:
                assert request_state.instance.id
                id_match = UnitxtMetric.ID_PATTERN.match(request_state.instance.id)
                assert id_match
                unitxt_split_name = id_match.group(1)
                row_index = int(id_match.group(2))
                references.append(dataset_dict[unitxt_split_name][row_index])
                assert request_state.result
                assert len(request_state.result.completions) == 1
                predictions.append(request_state.result.completions[0].text)

            # Compute metrics
            evaluate_results: List[Dict] = evaluate(predictions=predictions, dataset=references)

            # Extract instance metrics
            for request_state, evaluate_result in zip(request_states, evaluate_results):
                instance = request_state.instance
                instance_stats: List[Stat] = []
                instance_results = evaluate_result["score"]["instance"]
                for metric_name, metric_score in instance_results.items():
                    if metric_name == "score" or metric_name == "score_name":
                        continue
                    assert instance.split == helm_split
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
                            if isinstance(metric_score_element, numbers.Number):
                                stat = stat.add(metric_score_element)
                            else:
                                non_number_instance_metric_names.add(metric_name)
                    else:
                        if isinstance(metric_score, numbers.Number):
                            stat = stat.add(metric_score)
                        else:
                            non_number_instance_metric_names.add(metric_name)
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

            # Extract global metrics for this split
            if len(evaluate_results) > 0:
                global_results = evaluate_results[-1]["score"]["global"]
                for metric_name, metric_score in global_results.items():
                    if metric_name == "score" or metric_name == "score_name":
                        continue
                    stat = Stat(MetricName(name=metric_name, split=helm_split))
                    if isinstance(metric_score, list):
                        for metric_score_element in metric_score:
                            stat = stat.add(metric_score_element)
                    else:
                        stat = stat.add(metric_score)
                    aggregated_stats.append(stat)
        if non_number_instance_metric_names:
            hwarn(f"Ignored Unitxt instance metrics because they were not numbers: {non_number_instance_metric_names}")
        return MetricResult(aggregated_stats=aggregated_stats, per_instance_stats=per_instance_stats)

    def get_metadata(self):
        if len(self.kwargs) == 1 and "recipe" in self.kwargs:
            recipe = load_recipe(self.kwargs["recipe"])
        else:
            recipe = load_recipe(**self.kwargs)
        metric_names = recipe.task.metrics
        assert metric_names
        metric_metadata = []
        for metric_name in metric_names:
            metric, _ = fetch_artifact(metric_names[0])
            assert isinstance(metric, Metric)
            metric_name = metric.main_score
            metric_metadata.append(
                MetricMetadata(
                    name=metric_name,
                    display_name=metric_name,
                    description=f"{metric_name} (Unitxt)",
                    group="unitxt",
                ),
            )
        return metric_metadata
