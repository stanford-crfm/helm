from typing import Any, Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class OmniMATHMetric(Metric):
    """Score metrics for Omni-MATH."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        annotations: Dict[str, Any] = request_state.annotations["omni_math"]
        scores: List[int] = []
        for annotation_key, annotation_value in annotations.items():
            if annotation_key.endswith("_equivalence_judgement") and annotation_value is not None:
                scores.append(int(annotation_value))
        if not scores:
            raise ValueError("Could not compute Omni-MATH accuracy because all annotators failed.")
        score = sum(scores) / len(scores)
        return [
            Stat(MetricName("omni_math_accuracy")).add(score),
        ]
