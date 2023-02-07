from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat
from helm.common.human_task_request import HumanTaskTemplate, HumanQuestionTemplate, HumanTaskRequest


class HumanEvalSummarizationMetric(Metric):
    """
    Reimplementation of SummarizationMetric's human evals using Surge AI.
    """

    _NUM_WORKERS = 2

    _TEMPLATE = template = HumanTaskTemplate(
        name="HELM Summarization Evaluation v5",
        instructions="Rate the following summary.\n\nOriginal text:\n{{original_text}}\n\nSummary: {{summary}}",
        num_workers=2,
        questions=[
            HumanQuestionTemplate(
                question_type="multiple_choice",
                text="Can all the information expressed by the summary can be inferred from the article?",
                options=["Yes", "No"],
            ),
            HumanQuestionTemplate(
                question_type="multiple_choice",
                text="To what extend the summary include only important information from the source document? "
                "(1 = not at all, 5 = very much)",
                options=["1", "2", "3", "4", "5"],
            ),
            HumanQuestionTemplate(
                question_type="multiple_choice",
                text="Does the summary organize the relevant information into a well-structured summary "
                "(1 = not at all, 5 = very much)",
                options=["1", "2", "3", "4", "5"],
            ),
        ],
    )

    def __repr__(self):
        return "HumanEvalSummarizationMetric()"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Asks humans to rate a summarization."""
        assert request_state.result is not None
        if len(request_state.result.completions) != 1:
            raise ValueError("PoetryMetric only supports a single generation per instance")
        summary = request_state.result.completions[0].text
        request = HumanTaskRequest(
            self._TEMPLATE, fields={"original_text": request_state.instance.input.text, "summary": summary}
        )
        result = metric_service.get_human_task(request)
        if not result.answers:
            return []
        faithfulness_stat = Stat(MetricName("faithfulness"))
        for answer in result.answers[0]:
            faithfulness_stat.add(1 if answer == "yes" else 0)
        relevance_stat = Stat(MetricName("relevance"))
        for answer in result.answers[1]:
            relevance_stat.add((int(answer) - 1) / 4)
        coherence_stat = Stat(MetricName("coherence"))
        for answer in result.answers[2]:
            coherence_stat.add((int(answer) - 1) / 4)
        return [faithfulness_stat, relevance_stat, coherence_stat]
