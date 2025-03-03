from typing import Any, Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class HelpdeskCallSummarizationMetric(Metric):
    """Score metrics for helpdesk call summarization."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        annotations: Dict[str, Any] = request_state.annotations["helpdesk_call_center_summarization"]
        scores: List[int] = []
        for annotation_key, annotation_value in annotations.items():
            if annotation_key.endswith("_score") and annotation_value is not None:
                scores.append(annotation_value)
        if not scores:
            raise ValueError(
                "Could not compute score in HelpdeskCallSummarizationMetric because all annotators failed."
            )
        score = sum(scores) / len(scores)
        # normalize score
        score = (score - 1) / 9
        return [
            Stat(MetricName("call_summarization_score")).add(score),
        ]
