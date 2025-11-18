from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import compute_request_state_metrics
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class ArabicToolUsageMetric(Metric):
    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        assert "arabic_tool_usage" in request_state.annotations
        score = float(request_state.annotations["arabic_tool_usage"]["score"])
        return [Stat(MetricName("model_judged_score")).add(score)]
