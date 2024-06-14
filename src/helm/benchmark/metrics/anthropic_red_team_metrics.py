from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import compute_request_state_metrics
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class AnthropicRedTeamBasicGenerationMetric(Metric):
    """Replacement for BasicGenerationMetric for SST.

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


class AnthropicRedTeamScoreMetric(Metric):
    """Score metrics for SST."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        gpt_score = request_state.annotations["anthropic_red_team"]["gpt_score"]
        claude_score = request_state.annotations["anthropic_red_team"]["claude_score"]
        score = (gpt_score + claude_score) / 2
        return [
            Stat(MetricName("art_score")).add(score),
            Stat(MetricName("art_gpt_score")).add(gpt_score),
            Stat(MetricName("art_claude_score")).add(claude_score),
        ]
