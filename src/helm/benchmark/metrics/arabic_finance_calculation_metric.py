from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class ArabicFinanceCalculationMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        assert "arabic_finance_calculation" in request_state.annotations
        score = request_state.annotations["arabic_finance_calculation"]["score"]
        return [Stat(MetricName("calculation_accuracy")).add(score)]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="calculation_accuracy",
                display_name="Calculation accuracy",
                short_display_name="Accuracy",
                description="Fraction of instances that had final answer that was mathematically equivalent to the "
                "reference answer.",
                lower_is_better=False,
                group=None,
            ),
        ]
