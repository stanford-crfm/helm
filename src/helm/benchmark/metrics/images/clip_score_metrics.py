from statistics import mean
from typing import List

from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.window_services.clip_window_service import CLIPWindowService
from .clip_scorer import CLIPScorer
from .images_utils import is_blacked_out_image


class CLIPScoreMetric(Metric):
    """
    Defines metrics that use CLIPScore (https://arxiv.org/abs/2104.08718).
    CLIPScore is a reference free metric that can be used to evaluate the correlation between an image
    caption and the content of the image. It has been found to be highly correlated with human judgement.
    """

    def __init__(self):
        self._clip_scorer = CLIPScorer()

    def __repr__(self):
        return "CLIPScoreMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        request_result: RequestResult = request_state.result

        # Truncate the prompt using the CLIP tokenizer before feeding into the CLIP model.
        # Otherwise, the library will throw an error.
        prompt: str = CLIPWindowService(metric_service).truncate_from_right(request_state.request.prompt)

        scores: List[float] = []
        for image in request_result.completions:
            if image.file_path is not None and not is_blacked_out_image(image.file_path):
                clip_score: float = self._clip_scorer.compute_score(prompt, image.file_path)
                scores.append(clip_score)

        stats: List[Stat] = [
            Stat(MetricName("expected_clip_score")).add(mean(scores) if len(scores) > 0 else 0),
            Stat(MetricName("max_clip_score")).add(max(scores) if len(scores) > 0 else 0),
        ]
        return stats
