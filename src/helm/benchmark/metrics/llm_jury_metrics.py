from typing import Any, Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class LLMJuryMetric(Metric):
    """Score metrics for LLM Jury."""

    def __init__(
        self,
        metric_name: str,
        scenario_name: str,
        annotator_models: Dict[str, AnnotatorModelInfo],
        default_score: float = 0.0,
    ):
        self.metric_name = metric_name
        self.scenario_name = scenario_name
        self.annotator_models = annotator_models
        self.default_score = default_score

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        annotations: Dict[str, Any] = request_state.annotations[self.scenario_name]
        scores: List[int] = []
        score = self.default_score
        for annotation_key, annotation_dict in annotations.items():
            if annotation_key in self.annotator_models.keys() and annotation_dict is not None:
                for val in annotation_dict.values():
                    scores.append(int(val["score"]))
        if scores:
            score = sum(scores) / len(scores)
        return [
            Stat(MetricName(self.metric_name)).add(score),
        ]
