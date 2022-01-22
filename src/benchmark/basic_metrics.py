from typing import List, Callable

from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService


def exact_match(gold: str, pred: str) -> float:
    return 1 if gold == pred else 0


class BasicMetric(Metric):
    """
    Defines basic metrics which don't require domain knowledge.  This should be
    fairly comprehensive already and we should try to use this as much as possible.
    If we need a different variant, try to generalize this or factor things out.
    It's possible we don't need to subclass this.
    `names` is a list of optional metrics to be specified by the user. Currently only `exact_match` is supported.
    """

    def __init__(self, names: List[str]):
        self.names = names

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """
        Setup:
        - Gold (correct references): G1 ... Gm
        - Predictions (completions): P1 ... Pk

        For each pair (G, P), we can define a ${score} (e.g., exact match, F1, BLEU).

        We define the following stats:
        - ${score}: max_i score(Gi, P1)
        - ${score}@k: max_{i,j} score(Gi, Pj)
        """

        def get_num_bytes(text: str) -> int:
            """Compute the byte length of the input string"""
            return len(bytes(text, encoding="utf-8"))

        def compute_reference_metrics():
            """Compute the reference metrics."""

            def compute_metrics_helper(name: str, score_func: Callable[[str, str], float]) -> List[Stat]:
                score_1 = max(score_func(gold, preds[0]) for gold in golds)
                score_k = max(score_func(gold, pred) for gold in golds for pred in preds)

                return [
                    Stat(name).add(score_1),
                    Stat(f"{name}@{adapter_spec.num_outputs}").add(score_k),
                ]

            reference_metrics = []

            if "exact_match" in self.names:
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
                reference_metrics.extend(compute_metrics_helper("exact_match", exact_match))
            return reference_metrics

        def compute_language_modeling_metrics():
            """Compute the logprob and normalization factors for the first completion"""
            sequence = request_state.result.completions[0]
            logprob, num_tokens, num_bytes = sequence.logprob, len(sequence.tokens), get_num_bytes(sequence.text)

            # Ignore the conditioning prefix
            conditioning_prefix_length = 0
            conditioning_prefix_tokens = []
            for token in sequence.tokens:
                if conditioning_prefix_length >= len(adapter_spec.conditioning_prefix):
                    break
                conditioning_prefix_tokens.append(token)
                conditioning_prefix_length += len(token.text)
            assert "".join([token.text for token in conditioning_prefix_tokens]) == adapter_spec.conditioning_prefix

            logprob -= sum(token.logprob for token in conditioning_prefix_tokens)
            num_tokens -= len(conditioning_prefix_tokens)
            num_bytes -= get_num_bytes(adapter_spec.conditioning_prefix)

            return [Stat("logprob").add(logprob), Stat("num_tokens").add(num_tokens), Stat("num_bytes").add(num_bytes)]

        metrics = []

        if len(request_state.instance.references) > 0:
            metrics.extend(compute_reference_metrics())

        if request_state.result is not None:
            metrics.extend(compute_language_modeling_metrics())

        # Future: add F1, BLEU, etc.
        # TODO: pass in arguments to `BasicMetric`
        #       https://github.com/stanford-crfm/benchmarking/issues/44

        return metrics

    def evaluate_references(
        self, adapter_spec: AdapterSpec, reference_request_states: List[RequestState], metric_service: MetricService
    ) -> List[Stat]:
        """
        Setup: for each reference, we have a model score (log probability) and whether it's correct.
        We define the following metrics:
        - correct_rank: if we sort references by their logprobs, what is the ranking of the first correct reference.
        """
        # TODO: https://github.com/stanford-crfm/benchmarking/issues/45
        return []
