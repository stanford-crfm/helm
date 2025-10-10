from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class ALRAGEMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        assert "alrage" in request_state.annotations
        return [
            Stat(MetricName("alrage_score")).add(request_state.annotations["alrage"]["score"]),
        ]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="alrage_score",
                display_name="ALRAGE Score",
                short_display_name="Score",
                description="Score of the output judged by GPT-4o.",
                lower_is_better=False,
                group="accuracy",
            ),
        ]
