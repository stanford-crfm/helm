from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class WildBenchScoreMetric(Metric):
    """Score metrics for WildBench."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        all_scores = request_state.annotations["wildbench"]["score"]
        if len(all_scores) == 0:
            raise ValueError("Could not compute WB Score because all annotators failed.")
        score = sum(all_scores) / len(all_scores)
        score_rescaled = (score - 1) / 9
        return [
            Stat(MetricName("wildbench_score")).add(score),
            Stat(MetricName("wildbench_score_rescaled")).add(score_rescaled),
        ]
