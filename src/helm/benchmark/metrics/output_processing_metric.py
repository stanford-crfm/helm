import dataclasses
from typing import Any, Dict, List, TypedDict

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import (
    create_metric,
    Metric,
    MetricInterface,
    MetricResult,
    MetricSpec,
    PerInstanceStats,
)
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.object_spec import get_class_by_name
from helm.common.request import GeneratedOutput


class _MetricSpecDict(TypedDict):
    class_name: str
    args: Dict[str, Any]


def _dict_to_metric_spec(metric_spec_dict: _MetricSpecDict) -> MetricSpec:
    return MetricSpec(metric_spec_dict["class_name"], metric_spec_dict["args"])


class OutputProcessingMetric(MetricInterface):
    def __init__(self, processor: str, metric_specs: List[_MetricSpecDict]):
        self.processor = get_class_by_name(processor)  # actually a function, not a class
        self.metrics: List[Metric] = [create_metric(_dict_to_metric_spec(metric_spec)) for metric_spec in metric_specs]

    def _process_request_state(self, request_state: RequestState) -> RequestState:
        if not request_state.result:
            return request_state
        processed_completions: List[GeneratedOutput] = []
        for completion in request_state.result.completions:
            processed_completions.append(dataclasses.replace(completion, text=self.processor(completion.text)))
        return dataclasses.replace(
            request_state, result=dataclasses.replace(request_state.result, completions=processed_completions)
        )

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        aggregated_stats: List[Stat] = []
        per_instance_stats: List[PerInstanceStats] = []

        processed_scenario_state = dataclasses.replace(
            scenario_state,
            request_states=[
                self._process_request_state(request_state) for request_state in scenario_state.request_states
            ],
        )
        for metric in self.metrics:
            metric_result = metric.evaluate(processed_scenario_state, metric_service, eval_cache_path, parallelism)
            aggregated_stats.extend(metric_result.aggregated_stats)
            per_instance_stats.extend(metric_result.per_instance_stats)
        return MetricResult(aggregated_stats=aggregated_stats, per_instance_stats=per_instance_stats)
