from typing import List, Callable

from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric


class BasicMetric(Metric):
    """Produce basic metrics that just look at probabilities."""

    def evaluate_generation(self, adapter_spec: AdapterSpec, request_state: RequestState) -> List[Stat]:
        """
        Setup:
        - Gold (correct references): G1 ... Gm
        - Predictions (completions): P1 ... Pk

        For each pair (G, P), we can define a ${score} (e.g., exact match, F1, BLEU).

        We define the following stats:
        - ${score}: max_i score(Gi, P1)
        - ${score}@k: max_{i,j} score(Gi, Pj)
        """
        # Gold
        golds = [reference.output for reference in request_state.instance.references if reference.is_correct]
        assert len(golds) > 0

        # Prediction
        assert request_state.result is not None
        # TODO: Sort the predictions, or take them from the top tokens of the first completion
        preds = [completion.text for completion in request_state.result.completions]

        # Apply mapping if exists (e.g., for multiple-choice questions A -> Boston, B -> New York)

        def exact_match(gold: str, pred: str) -> float:
            return 1 if gold == pred else 0

        # TODO: add perplexity of the input text

        def compute_metrics(name: str, score_func: Callable[[str, str], float]) -> List[Stat]:
            score_1 = max(score_func(gold, preds[0]) for gold in golds)
            score_k = max(score_func(gold, pred) for gold in golds for pred in preds)

            return [
                Stat(name).add(score_1),
                Stat(f"{name}@{adapter_spec.num_outputs}").add(score_k),
            ]

        # Future: add F1, BLEU, etc.
        return compute_metrics("exact_match", exact_match)

    def evaluate_references(
        self, adapter_spec: AdapterSpec, reference_request_states: List[RequestState]
    ) -> List[Stat]:
        """
        Setup: for each reference, we have a model score (log probability) and whether it's correct.
        We define the following metrics:
        - ranking
        """
        # TODO
        return []
