"""Diversity metrics for the disinformation scenario."""

from typing import List

from sacrebleu.metrics import BLEU

from common.request import RequestResult
from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService


def _self_bleu(completions: List[str], **unused_kwargs) -> float:
    """Self-BLEU.

    Average over all scores, where each score is the BLEU of one generation compared against all other generations.
    """
    scores = []
    for i in range(len(completions)):
        hypothesis = completions[i]
        references = completions[:i] + completions[i + 1:]

        # Enable `effective_order` for sentence-level BLEU.
        score = BLEU(effective_order=True).sentence_score(hypothesis=hypothesis, references=references)
        scores.append(score.score)
    return sum(scores) / len(scores)


def _mauve(completions: List[str], references: List[str], **unused_kwargs) -> float:
    pass


metric_fns = {"self_bleu": _self_bleu, "mauve": _mauve}


class DisinformationMetric(Metric):
    def __init__(self, name):
        if name not in metric_fns:
            raise ValueError(f"Expected name to be one of {metric_fns.keys()}, but got {name}.")
        self._name = name
        self._metric_fn = metric_fns[name]

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        request_result: RequestResult = request_state.result
        completions = [completion.text.strip() for completion in request_result.completions]
        references = [reference.output.strip() for reference in request_state.instance.references]
        result = self._metric_fn(completions=completions, references=references)
        return [Stat(f"{self._name}").add(result)]
