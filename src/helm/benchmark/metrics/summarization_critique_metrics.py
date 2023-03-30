from typing import Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest


_FAITHFULNESS_NAME = "faithfulness"
_RELEVANCE_NAME = "relevance"
_COHERENCE_NAME = "coherence"
_YES_VALUE = "Yes"
_NO_VALUE = "No"


class SummarizationCritiqueMetric(Metric):
    """Reimplementation of SummarizationMetric's evals using critique evaluation.

    This is a demonstration of critique evaluation and is not indended for production use."""

    def __init__(self, num_respondants: int) -> None:
        self._template = CritiqueTaskTemplate(
            name="HELM Summarization Evaluation",
            # Note: Instructions can contain HTML.
            instructions="Rate the following summary.\n\nOriginal text:\n{{original_text}}\n\nSummary: {{summary}}",
            num_respondants=num_respondants,
            questions=[
                CritiqueQuestionTemplate(
                    name=_FAITHFULNESS_NAME,
                    question_type="multiple_choice",
                    # Note: Text can contain HTML.
                    text="Can all the information expressed by the summary can be inferred from the article?",
                    # Note: Options can contain HTML.
                    options=[_YES_VALUE, _NO_VALUE],
                ),
                CritiqueQuestionTemplate(
                    name=_RELEVANCE_NAME,
                    question_type="multiple_choice",
                    text="To what extend the summary include only important information from the source document?"
                    "(1 = not at all, 5 = very much)",
                    options=["1", "2", "3", "4", "5"],
                ),
                CritiqueQuestionTemplate(
                    name=_COHERENCE_NAME,
                    question_type="multiple_choice",
                    text="Does the summary organize the relevant information into a well-structured summary?"
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
        for response in result.responses:
            # TODO: This assume that every worker completes every task; check if this is true for Surge AI.
            for answer_name, answer in response.answers.items():
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
