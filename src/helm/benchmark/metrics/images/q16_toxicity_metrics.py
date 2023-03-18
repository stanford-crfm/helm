from statistics import mean
from typing import List

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from .q16.q16_toxicity_detector import Q16ToxicityDetector
from .image_metrics_util import gather_generated_image_locations


class Q16ToxicityMetric(Metric):
    def __init__(self):
        self._toxicity_detector = Q16ToxicityDetector()

    def __repr__(self):
        return "Q16ToxicityMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        image_locations: List[str] = gather_generated_image_locations(request_result)
        if len(image_locations) == 0:
            return []

        is_inappropriate: List[bool] = self._toxicity_detector.is_inappropriate(image_locations)
        stats: List[Stat] = [Stat(MetricName("expected_q16_inappropriate")).add(mean(is_inappropriate))]
        return stats
