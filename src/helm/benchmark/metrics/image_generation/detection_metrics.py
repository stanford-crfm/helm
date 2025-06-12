from typing import List, Dict, Any
import json
from statistics import mean

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.multimodal_request_utils import gather_generated_image_locations
from helm.benchmark.metrics.image_generation.detectors.vitdet import ViTDetDetector


class DetectionMetric(Metric):
    """
    Define metrics following DALL-EVAL (https://arxiv.org/abs/2202.04053),
    which measure whether generated images contain the correct objects, counts, and relations
    as specified in input text prompts.
    """

    def __init__(self):
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

        if self._detection_model is None:
            self._detection_model = ViTDetDetector()

        instance = request_state.instance
        references: Dict[str, Any] = {**json.loads(instance.references[0].output.text), "skill": instance.sub_split}

        prompt: str = request_state.request.prompt
        scores: List[float] = []
        for image_location in image_locations:
            score: float = self._detection_model.compute_score(prompt, image_location, references)
            scores.append(score)

        stats: List[Stat] = [
            Stat(MetricName("detection_correct_frac")).add(mean(scores) if len(scores) > 0 else 0),
        ]
        return stats
