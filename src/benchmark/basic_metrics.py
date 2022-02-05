from typing import List, Callable, Dict
from urllib.parse import unquote

from common.statistic import Stat
from common.request import Token
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService


def exact_match(gold: str, pred: str) -> float:
    return 1 if gold == pred else 0


def get_num_bytes(tokens: List[Token]) -> int:
    """Compute the byte length of the input tokens"""
    num_bytes = 0
    for token in tokens:
        if token.text.startswith("bytes:"):
            num_bytes += token.text.count("\\x")
        else:
            num_bytes += len(bytes(token.text, encoding="utf-8"))
    return num_bytes

def convert_tokens_to_text(tokens: List[Token]) -> List[Dict]:
    # Note: sometimes multiple tokens correspond to one character, for example:
    # ["bytes:\xe2\x80", "bytes:\x99"] => ’
    # For these, we keep these in the buffer and collapse them, and concatenate the entries.
    groups = []
    i = 0
    while i < len(tokens):
        # Aggregate consecutive tokens while they're "bytes:..."
        group = {"tokens": []}
        if (tokens[i].text.startswith('bytes:')):
            bytestring = ''
            while (i < len(tokens) and tokens[i].text.startswith('bytes:')):
                group["tokens"].append(tokens[i])
                # Extract part after : (e.g., \xe2\x80)
                bytestring += tokens[i].text.split(':')[1]
                i += 1
            # Convert to encoded URI (e.g., %e2%80%99) and decode
            group["text"] = unquote(bytestring.replace('\\x', '%'))
        else:
            group["tokens"].append(tokens[i])
            group["text"] = tokens[i].text
            i += 1
        groups.append(group)
    return groups

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

    def compute_reference_metrics(
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

    def compute_language_modeling_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the logprob and normalization factors for the first completion"""
        assert request_state.result is not None
        sequence = request_state.result.completions[0]
        assert "".join([group["text"] for group in convert_tokens_to_text(sequence.tokens)]) == request_state.request.prompt
        # try: # debug
        #     assert "".join([token.text for token in sequence.tokens]) == sequence.text
        # except:
        #     # print('#!#!#', "".join([token.text for token in sequence.tokens]), '!#!', sequence.text, '#!#!#')
        #     print('#!#!#', "".join([token.text for token in sequence.tokens]), '#!#!#')
        #     raise Exception("Assertion Error")

        pred_tokens = sequence.tokens[request_state.num_conditioning_tokens:]
        logprob, num_tokens, num_bytes = sum(token.logprob for token in pred_tokens), len(pred_tokens), get_num_bytes(pred_tokens)

        # Ignore the conditioning prefix
        # conditioning_prefix_tokens = sequence.tokens[: request_state.num_conditioning_tokens]
        # logprob -= sum(token.logprob for token in conditioning_prefix_tokens)
        # num_tokens -= len(conditioning_prefix_tokens)
        # num_bytes -= get_num_bytes("".join([token.text for token in conditioning_prefix_tokens]))  # TODO

        return [Stat("logprob").add(logprob), Stat("num_tokens").add(num_tokens), Stat("num_bytes").add(num_bytes)]

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the reference metrics and language modeling metrics"""
        metrics = []
        if len(request_state.instance.references) > 0:
            metrics.extend(self.compute_reference_metrics(adapter_spec, request_state, metric_service))

        metrics.extend(self.compute_language_modeling_metrics(adapter_spec, request_state, metric_service))

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
