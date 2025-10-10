from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class LiveQAScoreMetric(Metric):
    """Score metrics for LiveQA."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        score = request_state.annotations["live_qa"]["score"]
        return [Stat(MetricName("live_qa_score")).add(score)]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="live_qa_score",
                display_name="Judge Score",
                short_display_name=None,
                description="LLM-as-judge score",
                lower_is_better=False,
                group=None,
            ),
        ]
