from typing import Dict, List
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
from helm.common.request import RequestResult, Request, GeneratedOutput


class GPT4AudioCritiqueMetric(MetricInterface):
    """
    Critique evaluation for evaluating how original the generated text are given the image by GPT4.
    """

    # We can add more evaluation aspects here
    METRIC_NAME: str = "gpt4_audio_critique"
    GPT4_AUDIO_CRITIQUE_PROMPT: str = """###Task Description:
A ground truth answer, a response from a model to evaluate, and a score rubric \
representing a evaluation criteria are given.
1. Write a one-sentence feedback that assess the quality of the response \
strictly based on the given score rubric, not evaluating in general.
2. After writing the one-sentence feedback, write a score that is an \
integer between 1 and 5. You should refer to the score rubric.
3. Please do not generate any other opening, closing, and explanations.

###The ground truth answer:
{{ground_truth}}

###Model Response to evaluate:
{{orig_response}}

###Score Rubrics:
[Does the predicted response align with the ground truth \
in terms of accuracy?]

Score 1: The response is completely inaccurate or unrelated to the ground truth.
Score 2: The response contains significant inaccuracies or misinterpretations \
that distort the meaning of the ground truth.
Score 3: The response is mostly accurate but includes minor errors, omissions, \
or ambiguities.
Score 4: The response is accurate and aligns well with the ground truth, with \
only slight room for improvement.
Score 5: The response is fully accurate and precisely matches the ground truth \
with no errors or misinterpretations.

Your response should be in the format:
###Short Explanation: (explanation in only one sentence)
###Rating: (int)
"""

    def __init__(self, num_respondents: int, max_tokens: int):
        self._num_respondents = num_respondents
        self._max_tokens = max_tokens

    def __repr__(self) -> str:
        return "GPT4AudioCritiqueMetric()"

    def _extract_score_from_gpt_output(self, evaluator_response: str):
        """
        Extract the score from the evaluator response. Refer to the official Vibe-Eval implementation:
        https://github.com/reka-ai/reka-vibe-eval/blob/3852d4712da172a7b85dddeffc4f9c3482a6f4c9/evaluate.py#L159-#L164
        """
        re_match = re.search(r"Rating:\s*([1-5])", evaluator_response)
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
        input_request: Request = request_state.request
        # Predicted outputs and their originality scores
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        # Get input image and generated response for the originality evaluation
        assert input_request.multimodal_prompt is not None
        completions: List[GeneratedOutput] = request_result.completions
        generated_text: str = completions[0].text
        ref_text: str = request_state.instance.references[0].output.text

        template = CritiqueTaskTemplate(
            name="ahelm_gpt4_audio_critique",
            instructions=self.GPT4_AUDIO_CRITIQUE_PROMPT,
            num_respondents=self._num_respondents,
            max_tokens=self._max_tokens,
            questions=[
                CritiqueQuestionTemplate(
                    name=self.METRIC_NAME,
                    question_type=QuestionType.FREE_RESPONSE,
                    text="",
                    options=[],
                    # media_object=image_objects[0],  # we only take the first image as input
                )
            ],
        )

        request = CritiqueRequest(
            template=template,
            fields={
                "ground_truth": ref_text,
                "orig_response": generated_text,
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
                answer_value = self._extract_score_from_gpt_output(answer)
                stats[answer_name].add(answer_value)

        return list(stats.values())
