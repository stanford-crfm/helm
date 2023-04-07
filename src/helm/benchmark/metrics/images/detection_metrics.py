from typing import List

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from .image_metrics_utils import gather_generated_image_locations


class DetectionMetric(Metric):
    """
    TODO: write a short description of this metric here
    """

    def __init__(self):
        # TODO: if it takes a long to initialize, lazy load the model later (see clip_score_metrics.py as an example)
        self._detection_model = None

    def __repr__(self):
        return "DetectionMetric()"

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

        # TODO: At this point, we're evaluating multiple output images for a single prompt.
        #       Implement logic to evaluate (prompt, image) pairs. Examples:
        # https://github.com/stanford-crfm/helm/blob/vhelm/src/helm/benchmark/metrics/images/clip_score_metrics.py
        # https://github.com/stanford-crfm/helm/blob/vhelm/src/helm/benchmark/metrics/images/watermark_metrics.py

        prompt: str = request_state.request.prompt
        num_correct: int = 0
        for image_location in image_locations:
            if self._is_correct(prompt, image_location):
                num_correct += 1

        stats: List[Stat] = [
            Stat(MetricName("detection_correct_frac")).add(num_correct / len(image_locations)),
        ]
        return stats

    def _is_correct(self, prompt: str, image_location: str) -> bool:
        # TODO: Evaluate the prompt and image using the detection model
        return True
