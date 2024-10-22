from statistics import mean
from typing import List, Optional

from helm.common.images_utils import is_blacked_out_image
from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.image_generation.aesthetics_scorer import AestheticsScorer
from helm.common.multimodal_request_utils import gather_generated_image_locations


class AestheticsMetric(Metric):
    """
    Defines metrics for LAION's CLIP-based aesthetics predictor for images
    (https://github.com/LAION-AI/aesthetic-predictor).
    """

    def __init__(self):
        self._aesthetics_scorer: Optional[AestheticsScorer] = None

    def __repr__(self):
        return "AestheticsMetric()"

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

        if self._aesthetics_scorer is None:
            self._aesthetics_scorer = AestheticsScorer()

        # Compute the aesthetics score for each generated image. Skip blacked out images.
        scores: List[float] = [
            self._aesthetics_scorer.compute_aesthetics_score(location)
            for location in image_locations
            if not is_blacked_out_image(location)
        ]
        stats: List[Stat] = [
            Stat(MetricName("expected_aesthetics_score")).add(mean(scores) if len(scores) > 0 else 0),
            Stat(MetricName("max_aesthetics_score")).add(max(scores) if len(scores) > 0 else 0),
        ]
        return stats
