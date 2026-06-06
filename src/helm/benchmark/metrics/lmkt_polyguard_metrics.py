from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class PolyGuardMetric(Metric):
    """Score metrics for PolyGuard."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        scores = request_state.annotations["polyguard_autograder"]

        return [
            Stat(MetricName("harmful_response")).add(scores["harmful_response"]),
            Stat(MetricName("harmful_refusal")).add(scores["response_refusal"]),
        ]
