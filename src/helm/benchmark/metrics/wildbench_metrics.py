from typing import Any, Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class WildBenchScoreMetric(Metric):
    """Score metrics for WildBench."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        annotations: Dict[str, Any] = request_state.annotations["wildbench"]
        scores: List[float] = []
        for annotation_key, annotation_value in annotations.items():
            if annotation_key.endswith("_score") and annotation_value is not None:
                scores.append(annotation_value)
        if not scores:
            raise ValueError("Could not compute WB Score because all annotators failed.")
        score = sum(scores) / len(scores)
        score_rescaled = (score - 1) / 9
        return [
            Stat(MetricName("wildbench_score")).add(score),
            Stat(MetricName("wildbench_score_rescaled")).add(score_rescaled),
        ]
