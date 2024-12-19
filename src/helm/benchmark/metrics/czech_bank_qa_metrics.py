from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


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
