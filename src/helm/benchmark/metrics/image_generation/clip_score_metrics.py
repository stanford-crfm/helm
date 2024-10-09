from statistics import mean
from typing import List

from helm.common.general import singleton
from helm.common.request import RequestResult
from helm.common.clip_score_request import DEFAULT_CLIP_SCORE_MODEL, CLIPScoreResult, CLIPScoreRequest
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.common.images_utils import is_blacked_out_image
from helm.common.multimodal_request_utils import gather_generated_image_locations


class CLIPScoreMetric(Metric):
    """
    Defines CLIPScore-based metrics (https://arxiv.org/abs/2104.08718).
    CLIPScore is a reference free metric that can be used to evaluate the correlation between an image
    caption and the content of the image. It has been found to be highly correlated with human judgement.
    """

    def __init__(self, multilingual: bool = False):
        self._multilingual: bool = multilingual

    def __repr__(self):
        return f"CLIPScoreMetric(multilingual={self._multilingual})"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        def get_metric_name(base_name: str) -> str:
            if self._multilingual:
                base_name = f"{base_name}_multilingual"
            return base_name

        assert request_state.result is not None
        request_result: RequestResult = request_state.result

        prompt: str = request_state.request.prompt
        perturbation_name: str = request_state.instance.perturbation.name if request_state.instance.perturbation else ""
        if (
            request_state.instance.contrast_inputs is not None
            and len(request_state.instance.contrast_inputs) > 0
            and perturbation_name in ["translate", "dialect", "mild_mix"]
        ):
            prompt = singleton(request_state.instance.contrast_inputs).text

        # Truncate the prompt using the CLIP tokenizer before feeding into the CLIP model.
        # Otherwise, the library will throw an error.
        model = DEFAULT_CLIP_SCORE_MODEL
        prompt = WindowServiceFactory.get_window_service(model, metric_service).truncate_from_right(prompt)

        scores: List[float] = []
        image_locations: List[str] = gather_generated_image_locations(request_result)
        for location in image_locations:
            if not is_blacked_out_image(location):
                result: CLIPScoreResult = metric_service.compute_clip_score(
                    CLIPScoreRequest(prompt, location, model=model, multilingual=self._multilingual)
                )
                scores.append(result.score)

        stats: List[Stat] = [
            Stat(MetricName(get_metric_name("expected_clip_score"))).add(mean(scores) if len(scores) > 0 else 0),
            Stat(MetricName(get_metric_name("max_clip_score"))).add(max(scores) if len(scores) > 0 else 0),
        ]
        return stats
