from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import compute_request_state_metrics
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class MedicationQABasicGenerationMetric(Metric):
    """Replacement for BasicGenerationMetric for MedicationQA.

    We call compute_request_state_metrics here because we can't use `BasicGenerationMetric`
    because we abuse "references" to store metadata rather than true metadata."""

    def __init__(self):
        super().__init__()
        self.efficiency_metric = EfficiencyMetric()

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        return compute_request_state_metrics(self.efficiency_metric, adapter_spec, request_state, metric_service)


class MedicationQAScoreMetric(Metric):
    """Score metrics for MedicationQA."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        score = request_state.annotations["medication_qa"]["score"]
        return [Stat(MetricName("medication_qa_score")).add(score)]
