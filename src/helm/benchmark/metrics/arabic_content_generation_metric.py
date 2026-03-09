from typing import Any, Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class ArabicContentGenerationMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        scores: List[int] = []
        assert request_state.annotations
        annotations: Dict[str, Any] = request_state.annotations["arabic_content_generation"]
        for criteria_annotations in annotations.values():
            assert "score" in criteria_annotations
            scores.append(criteria_annotations["score"])
        if not scores:
            raise ValueError("ArabicContentGenerationMetric could not get scores from annotations.")
        score = sum(scores) / len(scores)
        score = (score - 1) / 4
        return [
            Stat(MetricName("arabic_content_generation_score")).add(score),
        ]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="arabic_content_generation_score",
                display_name="Arabic Content Generation Score",
                short_display_name="Score",
                description="LLM-judged quality score",
                lower_is_better=False,
                group="accuracy",
            ),
        ]
