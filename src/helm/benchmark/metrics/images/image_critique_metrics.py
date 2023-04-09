from typing import Dict, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.window_services.clip_window_service import CLIPWindowService
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest
from helm.common.images_utils import encode_base64
from helm.common.request import RequestResult
from .clip_scorers.base_clip_scorer import BaseCLIPScorer
from .clip_scorers.clip_scorer import CLIPScorer
from .image_metrics_utils import gather_generated_image_locations


class ImageCritiqueMetric(Metric):
    """Critique evaluation for image generation."""

    def __init__(
        self, include_alignment: bool, include_photorealism: bool, include_originality: bool, num_respondents: int
    ) -> None:
        questions: List[CritiqueQuestionTemplate] = []

        if include_alignment:
            questions.append(
                CritiqueQuestionTemplate(
                    name="image_text_alignment_human",
                    question_type="multiple_choice",
                    text="How well does the image match the description?",
                    options=[
                        "Does not match at all",
                        "Has significant discrepancies",
                        "Has several minor discrepancies",
                        "Has a few minor discrepancies",
                        "Matches exactly",
                    ],
                )
            )
        if include_photorealism:
            questions.append(
                CritiqueQuestionTemplate(
                    name="photorealism_human",
                    question_type="multiple_choice",
                    text="Does the image look like an AI-generated photo or a real photo?",
                    options=[
                        "AI-generated photo",
                        "Probably an AI-generated photo, but photorealistic",
                        "Neutral",
                        "Probably a real photo, but with irregular textures and shapes",
                        "Real photo",
                    ],
                )
            )
        if include_originality:
            questions.append(
                CritiqueQuestionTemplate(
                    name="originality_human",
                    question_type="multiple_choice",
                    text="How <u>original</u> is the image, given it was created with the description?",
                    options=[
                        "I’ve seen something like this before to the point it’s become tiresome.",
                        "The image is not really original, but it has some originality to it.",
                        "Neutral.",
                        "I find the image to be fresh and original.",
                        "I find the image to be extremely creative and out of this world.",
                    ],
                )
            )

        questions.extend(
            [
                CritiqueQuestionTemplate(
                    name="clear_subject_human",
                    question_type="multiple_choice",
                    text="Is it clear who the subject(s) of the image is?",
                    options=[
                        "Yes, it's clear.",
                        "No, it's unclear.",
                        "I don't know. It's hard to tell.",
                    ],
                ),
                CritiqueQuestionTemplate(
                    name="aesthetics_human",
                    question_type="multiple_choice",
                    text="How aesthetically pleasing is the image?",
                    options=[
                        "I find the image ugly.",
                        "The image has a lot of flaws, but it's not completely unappealing.",
                        "I find the image neither ugly nor aesthetically pleasing.",
                        "The image is aesthetically pleasing and nice to look at it.",
                        "The image is aesthetically stunning. I can look at it all day.",
                    ],
                ),
                CritiqueQuestionTemplate(
                    name="copyright_human",
                    question_type="free_response",
                    text="Please follow the instructions carefully:\n\n"
                    '1. Right click the image above and select "Search Image with Google”, which will open a '
                    "sidebar with Google Lens results.\n"
                    "2. Adjust the bounding box to fit the entire image if necessary.\n"
                    "3. Only for the first page of results, find possible copyright infringements by "
                    "looking for images that closely resemble the above image. For those images, "
                    "click on the image, which will open a new tab, and copy the URL for that tab.\n"
                    "4. List the URLs from step 3 below. If there are multiple URLs, list each on a new line.\n",
                    options=[],
                ),
            ]
        )

        self._template = CritiqueTaskTemplate(
            name="VHELM image evaluation",
            instructions="Please answer the questions below about the following image and description. "
            '\n<img src="data:image;base64,{{image}}">Description: <b>{{prompt}}</b>',
            num_respondents=num_respondents,
            questions=questions,
        )
        self._clip_scorer: Optional[BaseCLIPScorer] = None

    def __repr__(self) -> str:
        return "ImageCritiqueMetric()"

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

        # Use the CLIPScorer to select the best output image
        if self._clip_scorer is None:
            self._clip_scorer = CLIPScorer()
        prompt: str = CLIPWindowService(metric_service).truncate_from_right(request_state.request.prompt)
        image_path: str = self._clip_scorer.select_best_image(prompt, image_locations)

        # Send the critique request
        request = CritiqueRequest(self._template, fields={"prompt": prompt, "image": encode_base64(image_path)})
        result = metric_service.make_critique_request(request)
        if not result or not result.responses:
            return []

        # Skip computing metrics if there are not enough responses.
        if len(result.responses) < request.template.num_respondents:
            return []

        stats: Dict[str, Stat] = {}
        for question in self._template.questions:
            stats[question.name] = Stat(MetricName(question.name))

        for response in result.responses:
            for answer_name, answer in response.answers.items():
                answer_value: float

                # TODO: handle later -Tony
                if answer_name == "image_text_alignment_human":
                    if answer == "yes":
                        answer_value = 1
                    elif answer == "no":
                        answer_value = 0
                    else:
                        raise ValueError(f"Invalid answer {answer}")
                else:
                    raise ValueError(f"Invalid answer name: {answer_name}")

                stats[answer_name].add(answer_value)
        return list(stats.values())
