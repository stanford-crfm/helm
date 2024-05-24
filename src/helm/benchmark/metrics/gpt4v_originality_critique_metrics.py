from typing import Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats, add_context
from helm.benchmark.metrics.metric_name import MetricContext, MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest, QuestionType
from helm.common.hierarchical_logger import hlog
from helm.common.request import RequestResult, Request, GeneratedOutput
from helm.common.media_object import MultimediaObject, IMAGE_TYPE, MediaObject


class GPT4VCritiqueMetric(MetricInterface):
    """
    Critique evaluation for evaluating how original the generated text are given the image by GPT4V.
    """

    # We can add more evaluation aspects here
    ORIGINALITY_NAME: str = "originality_gpt4v"
    ORIGINALITY_ANSWER_TO_SCORE: Dict[str, int] = {
        "I’ve seen something like this before to the point it’s become tiresome.": 1,
        "The text is not really original, but it has some originality to it.": 2,
        "Neutral.": 3,
        "I find the text to be fresh and original.": 4,
        "I find the text to be extremely creative and out of this world.": 5,
    }

    def __init__(self, num_respondents: int):
        self._num_respondents = num_respondents

    def __repr__(self) -> str:
        return "GPT4CritiqueMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        request_states: List[RequestState] = scenario_state.request_states

        all_stats: Dict[MetricName, Stat] = {}
        per_instance_stats: List[PerInstanceStats] = []
        for request_state in request_states:
            context = MetricContext.from_instance(request_state.instance)
            stats_without_context = self.evaluate_generation(
                scenario_state.adapter_spec,
                request_state,
                metric_service,
                eval_cache_path,
            )
            stats = [add_context(stat_without_context, context) for stat_without_context in stats_without_context]
            for stat in stats:
                merge_stat(all_stats, stat)
            assert request_state.instance.id is not None
            per_instance_stats.append(
                PerInstanceStats(
                    instance_id=request_state.instance.id,
                    perturbation=request_state.instance.perturbation,
                    train_trial_index=request_state.train_trial_index,
                    stats=stats,
                )
            )
        return MetricResult(aggregated_stats=list(all_stats.values()), per_instance_stats=per_instance_stats)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        input_request: Request = request_state.request
        # Predicted outputs and their originality scores
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        # Get input image and generated response for the originality evaluation
        assert input_request.multimodal_prompt is not None
        completions: List[GeneratedOutput] = request_result.completions
        input_text: str = completions[0].text
        input_media: MultimediaObject = input_request.multimodal_prompt
        image_objects: List[MediaObject] = [
            item for item in input_media.media_objects if item.is_type(IMAGE_TYPE) and item.location
        ]

        template = CritiqueTaskTemplate(
            name="vhelm_gpt4v_originality",
            # TODO: Add proper instructions
            instructions="Answer the multiple choice question by just giving the letter of the correct "
            "answer.\n\n{{prompt}}",
            num_respondents=self._num_respondents,
            questions=[
                CritiqueQuestionTemplate(
                    name=self.ORIGINALITY_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    text="How original is the text, given it was created with the image?",
                    options=list(self.ORIGINALITY_ANSWER_TO_SCORE.keys()),
                    media_object=image_objects[0],  # we only take the first image as input
                )
            ],
        )
        request = CritiqueRequest(template=template, fields={"prompt": input_text})

        # send to critique request
        result = metric_service.make_critique_request(request)
        if not result or not result.responses:
            # Skip computing metrics if there aren't any responses yet
            hlog("Waiting for responses to be generated.")
            return []

        stats: Dict[str, Stat] = {}
        for question in template.questions:
            stats[question.name] = Stat(MetricName(question.name))

        for response in result.responses:
            for answer_name, answer in response.answers.items():
                assert isinstance(answer, str)
                answer_value: float
                answer_value = self.ORIGINALITY_ANSWER_TO_SCORE[answer]
                stats[answer_name].add(answer_value)

        return list(stats.values())
