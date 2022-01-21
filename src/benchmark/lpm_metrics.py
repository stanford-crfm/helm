from typing import List, Callable

from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric

def iou_match(gold: str, pred: str) -> float:
    pred = pred.split('\n')[0]
    if gold == "Nothing.":
        return float(pred == 'Nothing.')
    pred = pred.replace('.', '')
    gold = gold.replace('.', '')
    gold_set = set(gold.split(" is ")[-1].split(' and '))
    pred_set = set(pred.split(" is ")[-1].split(' and '))
    return (
        len(gold_set.intersection(pred_set)) / 
        len(gold_set.union(pred_set))
    )

def exact_set_match(gold: str, pred: str) -> float:
    pred = pred.split('\n')[0]
    if gold == "Nothing.":
        return float(pred == 'Nothing.')
    pred = pred.replace('.', '')
    gold = gold.replace('.', '')
    gold_set = set(gold.split(" is ")[-1].split(' and '))
    pred_set = set(pred.split(" is ")[-1].split(' and '))
    return float(gold_set == pred_set)

class LPMMetric(Metric):
    """
    Compare a predicted set to a target set
    """

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
        # Gold outputs
        golds = [reference.output for reference in request_state.instance.references if reference.is_correct]
        assert len(golds) > 0

        # Predicted outputs
        assert request_state.result is not None
        # TODO: Sort the predictions, or take them from the top tokens of the first completion
        #       https://github.com/stanford-crfm/benchmarking/issues/42
        preds = [completion.text.strip() for completion in request_state.result.completions]

        # Apply mapping if exists (e.g., for multiple-choice questions A -> Boston, B -> New York)
        if request_state.output_mapping is not None:
            preds = [request_state.output_mapping.get(pred) for pred in preds]

        # TODO: add perplexity of the input text
        #       https://github.com/stanford-crfm/benchmarking/issues/43

        def compute_metrics(name: str, score_func: Callable[[str, str], float]) -> List[Stat]:
            score_1 = max(score_func(gold, preds[0]) for gold in golds)
            score_k = max(score_func(gold, pred) for gold in golds for pred in preds)

            return [
                Stat(name).add(score_1),
                Stat(f"{name}@{adapter_spec.num_outputs}").add(score_k),
            ]

        # Future: add F1, BLEU, etc.
        # TODO: pass in arguments to `BasicMetric`
        #       https://github.com/stanford-crfm/benchmarking/issues/44
        return compute_metrics("iou_match", iou_match) + compute_metrics("exact_set_match", exact_set_match)

    def evaluate_references(
        self, adapter_spec: AdapterSpec, reference_request_states: List[RequestState]
    ) -> List[Stat]:
        """
        Setup: for each reference, we have a model score (log probability) and whether it's correct.
        We define the following metrics:
        - correct_rank: if we sort references by their logprobs, what is the ranking of the first correct reference.
        """
        # TODO: https://github.com/stanford-crfm/benchmarking/issues/45
        return []
