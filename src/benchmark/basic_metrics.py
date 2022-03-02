from typing import List, Callable, Optional

from common.general import format_tags
from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService
from proxy.tokenizer.auto_token_counter import AutoTokenCounter
from proxy.tokenizer.token_counter import TokenCounter


def exact_match(gold: str, pred: str) -> float:
    return 1 if gold == pred else 0


def get_num_bytes(text: str) -> int:
    """Compute the byte length of the input string"""
    return len(bytes(text, encoding="utf-8"))


def iou_set_match(gold: str, pred: str) -> float:
    """Compute the intersection over union of the gold and pred sets"""
    pred = pred.split("\n")[0]
    if gold == "Nothing.":
        return float(pred == "Nothing.")
    pred = pred.replace(".", "")
    gold = gold.replace(".", "")
    gold_set = set(gold.split(" is ")[-1].split(" and "))
    pred_set = set(pred.split(" is ")[-1].split(" and "))
    return len(gold_set.intersection(pred_set)) / len(gold_set.union(pred_set))


def exact_set_match(gold: str, pred: str) -> float:
    """Compute whether the sets generated exactly match"""
    pred = pred.split("\n")[0]
    if gold == "Nothing.":
        return float(pred == "Nothing.")
    pred = pred.replace(".", "")
    gold = gold.replace(".", "")
    gold_set = set(gold.split(" is ")[-1].split(" and "))
    pred_set = set(pred.split(" is ")[-1].split(" and "))
    return float(gold_set == pred_set)


class BasicMetric(Metric):
    """
    Defines basic metrics which don't require domain knowledge.  This should be
    fairly comprehensive already and we should try to use this as much as possible.
    If we need a different variant, try to generalize this or factor things out.
    It's possible we don't need to subclass this.
    `names` is a list of optional metrics to be specified by the user. Currently only `exact_match` is supported.
    """

    def __init__(self, names: List[str], group_tags: Optional[List[str]] = None):
        self.names: List[str] = names
        self.group_tags: List[str] = group_tags if group_tags else []
        self.token_counter: TokenCounter = AutoTokenCounter()

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

        def compute_metrics_helper(
            name: str, score_func: Callable[[str, str], float], tag: Optional[str] = None
        ) -> List[Stat]:
            score_1 = max(score_func(gold, preds[0]) for gold in golds)
            score_k = max(score_func(gold, pred) for gold in golds for pred in preds)

            group: str = format_tags([tag]) if tag else ""
            # TODO: clean this up once we have MetricNames
            #       https://github.com/stanford-crfm/benchmarking/issues/125
            return [
                Stat(f"{group + '_' if group else ''}{name}").add(score_1),
                Stat(f"{group + '_' if group else ''}{name}@{adapter_spec.num_outputs}").add(score_k),
            ]

        # maps each string metric name to its associated function
        metric_fn_mapping = {
            "exact_match": exact_match,
            "exact_set_match": exact_set_match,
            "iou_set_match": iou_set_match,
        }

        reference_metrics = []
        for metric_name in self.names:
            if metric_name in metric_fn_mapping:
                # Gold outputs
                golds = [reference.output for reference in request_state.instance.references if reference.is_correct]
                assert len(golds) > 0

                # Predicted outputs
                assert request_state.result is not None
                # TODO: Sort the predictions, or take them from the top tokens of the first completion
                #       https://github.com/stanford-crfm/benchmarking/issues/42
                preds = [completion.text for completion in request_state.result.completions]
                # import pdb; pdb.set_trace()
                # Apply mapping if exists (e.g., for multiple-choice questions A -> Boston, B -> New York)
                if request_state.output_mapping is not None:
                    preds = [request_state.output_mapping.get(pred) for pred in preds]
                reference_metrics.extend(compute_metrics_helper(metric_name, metric_fn_mapping[metric_name]))

                for group_tag in self.group_tags:
                    if group_tag in request_state.instance.tags:
                        reference_metrics.extend(
                            compute_metrics_helper(metric_name, metric_fn_mapping[metric_name], group_tag)
                        )
            else:
                raise NameError(f"{metric_name} is not in the list of metric functions.")
        return reference_metrics

    def compute_runtime_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute per-token normalized runtime"""
        assert request_state.result is not None

        runtime: float = request_state.result.request_time

        # Compute total number of tokens across completions
        num_tokens: int = sum([len(sequence.tokens) for sequence in request_state.result.completions])
        # Account for the tokens in prompt as well if echo_prompt is False
        if not request_state.request.echo_prompt:
            num_tokens_in_prompt: int = self.token_counter.tokenize_and_count(
                model=request_state.request.model, text=request_state.request.prompt
            )
            num_tokens += num_tokens_in_prompt

        return [Stat("runtime").add(runtime), Stat("normalized_runtime").add(runtime / num_tokens)]

    def compute_language_modeling_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the logprob and normalization factors for the first completion"""
        assert request_state.result is not None
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

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the reference metrics and language modeling metrics"""
        metrics = []
        if len(request_state.instance.references) > 0:
            metrics.extend(self.compute_reference_metrics(adapter_spec, request_state, metric_service))

        metrics.extend(self.compute_language_modeling_metrics(adapter_spec, request_state, metric_service))
        metrics.extend(self.compute_runtime_metrics(adapter_spec, request_state, metric_service))

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
