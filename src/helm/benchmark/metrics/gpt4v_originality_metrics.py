from typing import List

from helm.common.gpt4v_originality_request import (
    GPT4VOriginalityRequestResult,
)
from helm.common.request import Request, RequestResult, GeneratedOutput
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.common.media_object import MultimediaObject, MediaObject, IMAGE_TYPE
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat


class GPT4VOriginalityMetric(Metric):
    """
    Defines metrics for the originality evaluation based on GPT4V.
    """

    GPT4V_ORIGINALITY_MODEL_NAME: str = "openai/gpt-4-vision-preview"

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "GPT4VOriginalityMetric()"

    def _make_evaluation_content_from_multimedia(
        self, input_media: MultimediaObject, input_text: str
    ) -> MultimediaObject:
        """
        Seperates the image from the multimedia object and returns a new multimedia object with
        the image and the given text.
        """
        image_object = [item for item in input_media.media_objects if item.is_type(IMAGE_TYPE) and item.location]
        text_object = MediaObject(text=input_text, content_type="text/plain")
        return MultimediaObject(media_objects=[image_object[0], text_object])

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Given the proper prompt, we compute the orignality scores of VLM generated content
        given input image and the generated text.
        """
        request: Request = request_state.request
        # Predicted outputs and their originality scores
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        # Get input image and generated response for the originality evaluation
        assert request.multimodal_prompt is not None
        input_media: MultimediaObject = request.multimodal_prompt
        completions: List[GeneratedOutput] = request_result.completions

        input_text: str = completions[0].text
        evaluation_media: MultimediaObject = self._make_evaluation_content_from_multimedia(input_media, input_text)
        response: GPT4VOriginalityRequestResult = metric_service.get_gpt4v_originality_scores(
            request=Request(model=self.GPT4V_ORIGINALITY_MODEL_NAME, multimodal_prompt=evaluation_media)
        )
        if not response.success:
            raise Exception(f"Failed to get GPT4V originality scores: {response}")

        # Extract the originality scores from the response
        originality_scores: List[float] = [output.score for output in response.scores]
        num_originality_completions: int = len(originality_scores)

        max_originality_score: float = max(originality_scores) if len(originality_scores) > 0 else 0
        stats: List[Stat] = [
            Stat(MetricName("expected_max_originality")).add(max_originality_score),
            Stat(MetricName("originality_score")).add(sum(originality_scores) / num_originality_completions),
        ]

        return stats
