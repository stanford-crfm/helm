from typing import Dict, List

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest
from helm.common.images_utils import encode_base64, filter_blacked_out_images
from helm.common.request import RequestResult
from .image_metrics_utils import gather_generated_image_locations


class ImageCritiqueMetric(Metric):
    """
    Critique evaluation for image generation. Possesses the ability to ask human
    annotators the following questions about the generated images:

    1. Image-text alignment
    2. If the subject of the image is clear (for aesthetics)
    3. How aesthetically pleasing the image is?
    4. How original the image is?
    5. If there are any possible copyright infringements (originality)?
    """

    def __init__(
        self,
        study_title: str,
        include_alignment: bool,
        include_aesthetics: bool,
        include_originality: bool,
        include_copyright: bool,
        num_examples: int,
        num_respondents: int,
    ) -> None:
        self._study_title: str = study_title
        self._include_alignment: bool = include_alignment
        self._include_aesthetics: bool = include_aesthetics
        self._include_originality: bool = include_originality
        self._include_copyright: bool = include_copyright
        self._num_examples: int = num_examples
        self._num_respondents: int = num_respondents

    def __repr__(self) -> str:
        return "ImageCritiqueMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        request_states: List[RequestState] = scenario_state.request_states
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

        # Randomly select one of the generated images to critique
        image_path: str = np.random.choice(image_locations)
        prompt: str = request_state.request.prompt

        # Send the critique request
        template: CritiqueTaskTemplate = self._get_critique_template(adapter_spec.model)
        request = CritiqueRequest(template, fields={"prompt": prompt, "image": encode_base64(image_path)})
        result = metric_service.make_critique_request(request)
        if not result or not result.responses:
            return []

        # Skip computing metrics if there are not enough responses.
        if len(result.responses) < request.template.num_respondents:
            return []

        stats: Dict[str, Stat] = {}
        for question in template.questions:
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

    def _get_critique_template(self, model_name: str) -> CritiqueTaskTemplate:
        questions: List[CritiqueQuestionTemplate] = []

        if self._include_alignment:
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
        if self._include_originality:
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
        if self._include_aesthetics:
            questions.extend(
                [
                    CritiqueQuestionTemplate(
                        name="clear_subject_human",
                        question_type="multiple_choice",
                        text="Is it clear who the subject(s) of the image is? The subject can be a living being "
                        "(e.g., a dog or a person) or an inanimate body or object (e.g., a mountain).",
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
                ]
            )
        if self._include_copyright:
            questions.append(
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
                )
            )

        return CritiqueTaskTemplate(
            name=f"{self._study_title},{model_name}",
            instructions="Please answer the questions below about the following image and description. "
            '\n<img src="data:image;base64,{{image}}">Description: <b>{{prompt}}</b>',
            num_respondents=self._num_respondents,
            questions=questions,
        )
