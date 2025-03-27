from typing import Any, Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.medalign_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class MedalignMetric(Metric):
    """Score metrics for Medalign."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        annotations: Dict[str, Any] = request_state.annotations["medalign"]
        scores: List[int] = []
        score = 0.0
        for annotation_key, annotation_dict in annotations.items():
            if annotation_key in ANNOTATOR_MODELS.keys() and annotation_dict is not None:
                for val in annotation_dict.values():
                    scores.append(int(val["score"]))
        if scores:
            score = sum(scores) / len(scores)
        return [
            Stat(MetricName("medalign_accuracy")).add(score),
        ]
