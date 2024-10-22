from typing import Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest, QuestionType


_FAITHFULNESS_NAME = "faithfulness"
_RELEVANCE_NAME = "relevance"
_COHERENCE_NAME = "coherence"
_YES_VALUE = "Yes"
_NO_VALUE = "No"


class SummarizationCritiqueMetric(Metric):
    """Reimplementation of SummarizationMetric's evals using critique evaluation.

    This is a demonstration of critique evaluation and is not intended for production use."""

    def __init__(self, num_respondents: int) -> None:
        self._template = CritiqueTaskTemplate(
            name="summarization_critique",
            # Note: Instructions can contain HTML.
            # Note: To render new lines in any HTML block, you must use <p></p>, <br>, or style="white-space: pre-wrap;"
            instructions="<p>Rate the following summary.</p>"
            "<h4>Original Text</h4>"
            '<p style="white-space: pre-wrap;">{{original_text}}</p>'
            "<h4>Summary</h4>"
            '<p style="white-space: pre-wrap;">{{summary}}</p>',
            num_respondents=num_respondents,
            questions=[
                CritiqueQuestionTemplate(
                    name=_FAITHFULNESS_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    # Note: Text can contain HTML.
                    text="Can all the information expressed by the summary can be inferred from the article? ",
                    # Note: Options can contain HTML.
                    options=[_YES_VALUE, _NO_VALUE],
                ),
                CritiqueQuestionTemplate(
                    name=_RELEVANCE_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    text="To what extent the summary include only important information from the source document? "
                    "(1 = not at all, 5 = very much)",
                    options=["1", "2", "3", "4", "5"],
                ),
                CritiqueQuestionTemplate(
                    name=_COHERENCE_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    text="Does the summary organize the relevant information into a well-structured summary? "
                    "(1 = not at all, 5 = very much)",
                    options=["1", "2", "3", "4", "5"],
                ),
            ],
        )

    def __repr__(self) -> str:
        return "SummarizationCritiqueMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Get critiques of a summary and compute metrics based on the critiques."""
        assert request_state.result is not None
        if len(request_state.result.completions) != 1:
            raise ValueError("SummarizationCritiqueMetric only supports a single generation per instance")
        summary = request_state.result.completions[0].text
        request = CritiqueRequest(
            self._template, fields={"original_text": request_state.instance.input.text, "summary": summary}
        )
        result = metric_service.make_critique_request(request)
        if not result or not result.responses:
            return []
        stats: Dict[str, Stat] = {}
        for question in self._template.questions:
            stats[question.name] = Stat(MetricName(question.name))

        # Skip computing metrics if there are not enough responses.
        if len(result.responses) < request.template.num_respondents:
            return []

        for response in result.responses:
            for answer_name, answer in response.answers.items():
                if not isinstance(answer, str):
                    raise ValueError(f"Expected answer to {answer_name} be a string")
                answer_value: float
                if answer_name == _FAITHFULNESS_NAME:
                    if answer == _YES_VALUE:
                        answer_value = 1
                    elif answer == _NO_VALUE:
                        answer_value = 0
                    else:
                        raise ValueError(f"Unknown answer to {_FAITHFULNESS_NAME}: {answer}")
                else:
                    answer_value = (int(answer) - 1) / 4
                stats[answer_name].add(answer_value)
        return list(stats.values())
