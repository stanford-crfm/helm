from typing import List

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest
from helm.common.images_utils import encode_base64, filter_blacked_out_images
from helm.common.request import RequestResult
from .image_metrics_utils import gather_generated_image_locations
from ...scenarios.scenario import Reference


class PhotorealismCritiqueMetric(Metric):
    """
    Critique evaluation for evaluating how photorealistic the generated images are.
    """

    AI_GENERATED_PHOTO_RESPONSE: str = "AI-generated photo"
    REAL_PHOTO_RESPONSE: str = "Real photo"

    def __init__(self, num_examples: int, num_respondents: int) -> None:
        self._num_examples: int = num_examples
        self._num_respondents: int = num_respondents

    def __repr__(self) -> str:
        return "PhotorealismCritiqueMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        request_states: List[RequestState] = scenario_state.request_states
        np.random.seed(0)
        request_states = list(
            np.random.choice(
                request_states,  # type: ignore
                self._num_examples,
                replace=False,
            )
        )

        questions: List[CritiqueQuestionTemplate] = []
        for request_state in request_states:
            assert request_state.result is not None
            request_result: RequestResult = request_state.result
            image_locations: List[str] = gather_generated_image_locations(request_result)
            image_locations = filter_blacked_out_images(image_locations)
            if len(image_locations) == 0:
                return MetricResult(aggregated_stats=[], per_instance_stats=[])

            # Randomly select one of the generated images to critique and real image to compare to
            generated_image_path: str = np.random.choice(image_locations)
            references: List[Reference] = request_state.instance.references
            assert len(references) > 0, "Need at least one reference image for this metric"
            selected_reference: Reference = np.random.choice(references)  # type: ignore
            assert selected_reference.output.file_path is not None
            real_image_path: str = selected_reference.output.file_path

            for image_path, correct_response in [
                (generated_image_path, self.AI_GENERATED_PHOTO_RESPONSE),
                (real_image_path, self.REAL_PHOTO_RESPONSE),
            ]:
                base64_image: str = encode_base64(image_path)
                questions.append(
                    CritiqueQuestionTemplate(
                        name="photorealism_human",
                        question_type="multiple_choice",
                        text="Does the image look like an AI-generated photo or a real photo? "
                        f'\n<img src="data:image;base64,{base64_image}">',
                        options=[
                            self.AI_GENERATED_PHOTO_RESPONSE,
                            "Probably an AI-generated photo, but photorealistic",
                            "Neutral",
                            "Probably a real photo, but with irregular textures and shapes",
                            self.REAL_PHOTO_RESPONSE,
                        ],
                        correct_option=correct_response,
                    )
                )

        # Randomly shuffle questions to get a mix of real and generated images
        np.random.shuffle(questions)  # type: ignore
        template = CritiqueTaskTemplate(
            name=f"VHELM image evaluation - photorealism,{scenario_state.adapter_spec.model}",
            instructions="Determine if the following images are AI-generated or real.",
            num_respondents=self._num_respondents,
            questions=questions,
        )

        # Send the critique request
        request = CritiqueRequest(template, fields={})
        result = metric_service.make_critique_request(request)
        if not result or not result.responses:
            return MetricResult(aggregated_stats=[], per_instance_stats=[])

        # Skip computing metrics if there are not enough responses.
        if len(result.responses) < request.template.num_respondents:
            return MetricResult(aggregated_stats=[], per_instance_stats=[])

        # TODO: compute score -Tony
        score: float = 0
        return MetricResult(aggregated_stats=[Stat(MetricName("photorealism_human")).add(score)], per_instance_stats=[])
