from typing import Dict, List, Optional
import re

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import MetricInterface, MetricResult, PerInstanceStats, add_context
from helm.benchmark.metrics.metric_name import MetricContext, MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest, QuestionType
from helm.common.hierarchical_logger import hlog
from helm.common.request import RequestResult, GeneratedOutput
from helm.common.media_object import MultimediaObject, IMAGE_TYPE, MediaObject, TEXT_TYPE


class PrometheusVisionCritiqueMetric(MetricInterface):
    """
    We compute the same metrics from the Prometheus-Vision: Vision-Language Model as a Judge for
    Fine-Grained Evaluation paper:
    https://arxiv.org/pdf/2401.06591.pdf

    In this paper, the output of a Vision-Language Model named Prometheus-Vision is used to evaluate
    the quality of the output of other Vision-Language Models to be evaluated.
    """

    # We can add more evaluation aspects here
    METRIC_NAME: str = "prometheus_vision"
    METRIC_PROMPT: str = """A chat between a curious human and an artificial intelligence assistant. \
        The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:<image>\
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, \
    image and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not \
    evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: Feedback: (write a feedback for criteria) [RESULT] (an integer number \
    between 1 and 5)
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{{orig_instruction}}

###Response to evaluate:
{{orig_response}}

###Reference Answer (Score 5):
{{orig_reference_answer}}

###Score Rubrics:
[{{orig_criteria}}]
Score 1: {{orig_score1_description}}
Score 2: {{orig_score2_description}}
Score 3: {{orig_score3_description}}
Score 4: {{orig_score4_description}}
Score 5: {{orig_score5_description}}

###Feedback:
ASSISTANT:
"""

    def __init__(self, num_respondents: int, max_tokens: int):
        self._num_respondents = num_respondents
        self._max_tokens = max_tokens

    def __repr__(self) -> str:
        return "PrometheusVisionCritiqueMetric()"

    def _extract_score_from_prometheus_vision_output(self, evaluator_response: str):
        evaluator_response = evaluator_response.split("ASSISTANT:")[1]
        re_match = re.search(r"\s*([1-5])", evaluator_response)
        if re_match is None:
            hlog(f"Error parsing answer: {evaluator_response}. Skipping question (and so the respondent entirely)")
            return None
        return int(re_match.group(1))

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
        input_content = request_state.request
        # Predicted outputs and their prometheus vision scores
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        # Get input image and generated response for evaluation
        assert input_content.multimodal_prompt is not None
        completions: List[GeneratedOutput] = request_result.completions
        generated_text: str = completions[0].text
        input_media: MultimediaObject = input_content.multimodal_prompt
        ref_text: str = request_state.instance.references[0].output.text
        image_objects: List[MediaObject] = [
            item for item in input_media.media_objects if item.is_type(IMAGE_TYPE) and item.location
        ]
        input_text: Optional[str] = [item for item in input_media.media_objects if item.is_type(TEXT_TYPE)][0].text

        template = CritiqueTaskTemplate(
            name="vhelm_prometheus_vision",
            instructions=self.METRIC_PROMPT,
            num_respondents=self._num_respondents,
            max_tokens=self._max_tokens,
            questions=[
                CritiqueQuestionTemplate(
                    name=self.METRIC_NAME,
                    question_type=QuestionType.FREE_RESPONSE,
                    text="",
                    options=[],
                    media_object=image_objects[0],  # we only take the first image as input
                )
            ],
        )
        request = CritiqueRequest(
            template=template,
            fields={
                "orig_instruction": input_text if input_text is not None else "",
                "orig_response": generated_text,
                "orig_reference_answer": ref_text,
                "orig_criteria": "similarity between the reponse and the reference.",
                "orig_score1_description": "The model's responses do not follow the instructions provided.",
                "orig_score2_description": "The resulting response follows the instructions, but the answer \
                    is completely wrong relative to the reference answer.",
                "orig_score3_description": "The resulting response follows the instructions, but the answer is \
                    partially wrong relative to the reference answer.",
                "orig_score4_description": "The resulting response follows the instructions, the overall answer \
                    is relatively perfect with only a very few errors.",
                "orig_score5_description": "The overall answer is completely correct compared to the reference \
                    answer, and conforms to the instructions provided.",
            },
        )
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
                answer_value = self._extract_score_from_prometheus_vision_output(answer)
                stats[answer_name].add(answer_value)

        return list(stats.values())
