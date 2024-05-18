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
import re


class PrometheusVisionMetric(MetricInterface):
    """
    We compute the same metrics from the Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation paper:
    https://arxiv.org/pdf/2401.06591.pdf

    In this paper, the output of a Vision-Language Model named Prometheus-Vision is used to evaluate the quality of the output of other Vision-Language Models to be evaluated.
    """

    # We can add more evaluation aspects here
    METRIC_NAME: str = "prometheus_vision"
    METRIC_PROMPT: str = (
        """###Task Description:\n An instruction (might include an Input inside it), a response to evaluate and image. 1. Write a detailed feedback that assesses the quality of the response, not evaluating in general. 2. After writing a feedback, write a score that is an integer between 1 and 5. 3. The output format should look as follows: Feedback, [RESULT] (an integer number between 1 and 5) 4. Please do not generate any other opening, closing, and explanations.\n\n###The instruction to evaluate: {{instruction}}\n\n"""
    )
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
        return "PrometheusVisionMetric()"

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
        # Predicted outputs and their prometheus vision scores
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        # Get input image and generated response for evaluation
        assert input_request.multimodal_prompt is not None
        completions: List[GeneratedOutput] = request_result.completions
        input_text: str = completions[0].text
        input_media: MultimediaObject = input_request.multimodal_prompt
        image_object: List[MediaObject] = [
            item for item in input_media.media_objects if item.is_type(IMAGE_TYPE) and item.location
        ]
        image_url = image_object[0].location  # we only take the first image as input

        template = CritiqueTaskTemplate(
            name="vhelm_prometheus_vision",
            instructions="Answer the question given the text and image." "\n\n{{prompt}}",
            num_respondents=self._num_respondents,
            questions=[
                CritiqueQuestionTemplate(
                    name=self.METRIC_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    text=self.METRIC_PROMPT,
                    options=list(self.ORIGINALITY_ANSWER_TO_SCORE.keys()),
                    image_url=image_url,
                )
            ],
        )
        request = CritiqueRequest(template=template, fields={"prompt": input_text, "instruction": input_text})

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
                answer_value = float(re.findall(r"\d+", answer)[0])
                stats[answer_name].add(answer_value)

        return list(stats.values())
