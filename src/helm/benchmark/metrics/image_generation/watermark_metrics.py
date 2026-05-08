from statistics import mean
from typing import List

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.multimodal_request_utils import gather_generated_image_locations
from helm.benchmark.metrics.image_generation.watermark.watermark_detector import WatermarkDetector


class WatermarkMetric(Metric):
    """
    Defines metrics for detecting watermarks in images using the
    LAION's watermark detector (https://github.com/LAION-AI/LAION-5B-WatermarkDetection).
    """

    def __init__(self):
        self._watermark_detector = WatermarkDetector()

    def __repr__(self):
        return "WatermarkMetric()"

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

        # Batch process the images and detect if they have watermarks
        has_watermarks, watermark_probs = self._watermark_detector.has_watermark(image_locations)
        stats: List[Stat] = [
            Stat(MetricName("watermark_frac")).add(mean(has_watermarks) if len(has_watermarks) > 0 else 0),
            Stat(MetricName("expected_max_watermark_prob")).add(
                max(watermark_probs) if len(watermark_probs) > 0 else 0
            ),
        ]
        return stats

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="watermark_frac",
                display_name="Watermark frac",
                short_display_name="Watermark frac",
                description="Watermark detector from LAION to determine whether an image contains watermarks.",
                lower_is_better=True,
                group="heim_originality_watermark_metrics",
            ),
            MetricMetadata(
                name="expected_max_watermark_prob",
                display_name="Expected maximum watermark prob",
                short_display_name="Expected max watermark prob",
                description="Watermark detector from LAION to determine whether an image contains watermarks.",
                lower_is_better=True,
                group=None,
            ),
        ]
