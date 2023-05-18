from typing import Dict, List

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.benchmark.scenarios.scenario import Reference
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest, QuestionType
from helm.common.images_utils import encode_base64, filter_blacked_out_images
from helm.common.hierarchical_logger import hlog
from helm.common.request import RequestResult
from .image_metrics_utils import gather_generated_image_locations


class PhotorealismCritiqueMetric(Metric):
    """
    Critique evaluation for evaluating how photorealistic the generated images are.
    """

    PHOTOREALISM_NAME: str = "photorealism_human"
    PHOTOREALISM_ANSWER_TO_SCORE: Dict[str, int] = {
        "AI-generated photo": 1,
        "Probably an AI-generated photo, but photorealistic": 2,
        "Neutral": 3,
        "Probably a real photo, but with irregular textures and shapes": 4,
        "Real photo": 5,
    }

    def __init__(self, num_examples: int, num_respondents: int, use_perturbed: bool = False) -> None:
        self._num_examples: int = num_examples
        self._num_respondents: int = num_respondents
        self._use_perturbed: bool = use_perturbed

    def __repr__(self) -> str:
        return "PhotorealismCritiqueMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        request_states: List[RequestState] = []
        if self._use_perturbed:
            for request_state in scenario_state.request_states:
                if request_state.instance.perturbation is not None:
                    request_states.append(request_state)
        else:
            request_states = scenario_state.request_states

        np.random.seed(0)
        if self._num_examples < len(request_states):
            request_states = list(
                np.random.choice(
                    request_states,  # type: ignore
                    self._num_examples,
                    replace=False,
                )
            )

        all_stats: Dict[MetricName, Stat] = {}
        for request_state in request_states:
            stats = self.evaluate_generation(
                scenario_state.adapter_spec,
                request_state,
                metric_service,
                eval_cache_path,
            )
            for stat in stats:
                merge_stat(all_stats, stat)

        return MetricResult(aggregated_stats=list(all_stats.values()), per_instance_stats=[])

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
        image_locations = filter_blacked_out_images(image_locations)
        if len(image_locations) == 0:
            return []

        # Randomly select one of the generated images to critique and real image to compare to
        generated_image_path: str = np.random.choice(image_locations)
        references: List[Reference] = request_state.instance.references
        assert len(references) > 0, "Need at least one reference image for this metric"
        selected_reference: Reference = np.random.choice(references)  # type: ignore
        assert selected_reference.output.file_path is not None
        real_image_path: str = selected_reference.output.file_path

        template = CritiqueTaskTemplate(
            # name=f"VHELM photorealism,{scenario_state.adapter_spec.model}",
            name="vhelm_photorealism",
            instructions="<p>Determine if the following image is AI-generated or real.</p>"
            '<br><img src="data:image;base64,{{image}}"><br>',
            num_respondents=self._num_respondents,
            questions=[
                CritiqueQuestionTemplate(
                    name=self.PHOTOREALISM_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    text="Does the image look like an AI-generated photo or a real photo?",
                    options=list(self.PHOTOREALISM_ANSWER_TO_SCORE.keys()),
                )
            ],
        )

        generated_stat = Stat(MetricName("photorealism_generated_human"))
        real_stat = Stat(MetricName("photorealism_real_human"))

        for image_path, stat in [(generated_image_path, generated_stat), (real_image_path, real_stat)]:
            request = CritiqueRequest(template, fields={"image": encode_base64(image_path)})
            result = metric_service.make_critique_request(request)
            if not result or len(result.responses) == 0:
                # Skip computing metrics if there aren't any responses yet
                hlog("Waiting for responses to be collected.")
                continue

            for response in result.responses:
                answer: str = str(response.answers[self.PHOTOREALISM_NAME])
                score: float = self.PHOTOREALISM_ANSWER_TO_SCORE[answer]
                stat.add(score)

        return [generated_stat, real_stat]
