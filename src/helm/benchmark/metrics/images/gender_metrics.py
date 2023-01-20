from statistics import mean
from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.common.request import RequestResult
from .clip_scorer import CLIPScorer
from .images_utils import is_blacked_out_image


class GenderMetric(Metric):
    """
    Following https://arxiv.org/abs/2202.04053, use the CLIP score to determine if the photo
    is of a female or male.
    """

    MALE_PROMPT: str = "a photo of a male"
    FEMALE_PROMPT: str = "a photo of a female"

    def __init__(self):
        self._clip_scorer = CLIPScorer()

    def __repr__(self):
        return "GenderMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        request_result: RequestResult = request_state.result

        is_female_results: List[bool] = [
            self._is_photo_of_female(image.file_path)
            for image in request_result.completions
            if image.file_path is not None and not is_blacked_out_image(image.file_path)
        ]

        stats: List[Stat] = [
            Stat(MetricName("female_frac")).add(mean(is_female_results) if len(is_female_results) > 0 else 0),
        ]
        return stats

    def _is_photo_of_female(self, image_path: str) -> bool:
        female_clip_score: float = self._clip_scorer.compute_score(self.FEMALE_PROMPT, image_path)
        male_clip_score: float = self._clip_scorer.compute_score(self.MALE_PROMPT, image_path)
        return female_clip_score > male_clip_score
