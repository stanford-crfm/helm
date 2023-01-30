from statistics import mean
from typing import List, Optional

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.common.request import RequestResult
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.window_services.clip_window_service import CLIPWindowService
from helm.common.images_utils import is_blacked_out_image
from helm.common.gpu_utils import empty_cuda_cache
from .clip_scorers.clip_scorer import CLIPScorer
from .clip_scorers.multilingual_clip_scorer import MultilingualCLIPScorer


class CLIPScoreMetric(Metric):
    """
    Defines metrics that use CLIPScore (https://arxiv.org/abs/2104.08718).
    CLIPScore is a reference free metric that can be used to evaluate the correlation between an image
    caption and the content of the image. It has been found to be highly correlated with human judgement.
    """

    def __init__(self):
        self._clip_scorer: Optional[CLIPScorer] = None
        self._multilingual_clip_scorer: Optional[MultilingualCLIPScorer] = None

    def __repr__(self):
        return "CLIPScoreMetric()"

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        result: MetricResult = super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism)

        # Free up GPU memory
        del self._clip_scorer
        del self._multilingual_clip_scorer
        empty_cuda_cache()

        return result

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        request_result: RequestResult = request_state.result

        if self._clip_scorer is None:
            self._clip_scorer = CLIPScorer()
        if self._multilingual_clip_scorer is None:
            self._multilingual_clip_scorer = MultilingualCLIPScorer()

        # Truncate the prompt using the CLIP tokenizer before feeding into the CLIP model.
        # Otherwise, the library will throw an error.
        prompt: str = CLIPWindowService(metric_service).truncate_from_right(request_state.request.prompt)

        scores: List[float] = []
        multilingual_clip_scores: List[float] = []

        for image in request_result.completions:
            if image.file_path is not None and not is_blacked_out_image(image.file_path):
                clip_score: float = self._clip_scorer.compute_score(prompt, image.file_path)
                scores.append(clip_score)

                clip_score = self._multilingual_clip_scorer.compute_score(prompt, image.file_path)
                multilingual_clip_scores.append(clip_score)

        stats: List[Stat] = [
            Stat(MetricName("expected_clip_score")).add(mean(scores) if len(scores) > 0 else 0),
            Stat(MetricName("max_clip_score")).add(max(scores) if len(scores) > 0 else 0),
            Stat(MetricName("expected_clip_score_multilingual")).add(
                mean(multilingual_clip_scores) if len(multilingual_clip_scores) > 0 else 0
            ),
            Stat(MetricName("max_clip_score_multilingual")).add(
                max(multilingual_clip_scores) if len(multilingual_clip_scores) > 0 else 0
            ),
        ]
        return stats
