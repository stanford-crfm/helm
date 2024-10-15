from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import compute_request_state_metrics
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


# class AIRBench2024BasicGenerationMetric(Metric):
#     """Replacement for BasicGenerationMetric for AIRBench 2024.

#     We call compute_request_state_metrics here because we can't use `BasicGenerationMetric`
#     because we abuse "references" to store metadata rather than true metadata."""

#     def __init__(self):
#         super().__init__()
#         self.efficiency_metric = EfficiencyMetric()

#     def evaluate_generation(
#         self,
#         adapter_spec: AdapterSpec,
#         request_state: RequestState,
#         metric_service: MetricService,
#         eval_cache_path: str,
#     ) -> List[Stat]:
#         return compute_request_state_metrics(self.efficiency_metric, adapter_spec, request_state, metric_service)


class CzechBankQAMetrics(Metric):
    """Score metrics for AIRBench 2024."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # assert len(request_state.instance.references) > 1
        # category_text = request_state.instance.references[0].output.text
        # category_parts = category_text.split(".")
        # assert len(category_parts) == 3
        assert request_state.annotations
        error_rate = 0.0 if request_state.annotations["czech_bank_qa"]["error"] is None else 1.0
        return [
            Stat(MetricName("error_rate")).add(error_rate),
        ]
