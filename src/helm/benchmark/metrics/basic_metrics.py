from collections import defaultdict
import math
from dataclasses import dataclass
from typing import List, Dict, Set
from urllib.parse import unquote

import numpy as np
import scipy  # type: ignore
import calibration as cal  # type: ignore
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.evaluate_reference_metrics import compute_reference_metrics
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric
from helm.benchmark.metrics.reference_metric import ReferenceMetric

from helm.common.hierarchical_logger import hlog
from helm.common.request import Token, GeneratedOutput
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    ADAPT_RANKING_BINARY,
)
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.window_services.window_service import WindowService
from helm.benchmark.window_services.window_service_factory import WindowServiceFactory
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Instance
from helm.benchmark.metrics.metric import Metric, MetricInterface, MetricResult, add_context, get_unique_stat_by_name
from helm.benchmark.metrics.metric_name import MetricContext, MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat


def get_num_bytes(tokens: List[Token]) -> int:
    """
    Compute the byte length of the input tokens. For a UTF-8 string token, we use byte() to convert
    it to bytes; for byte tokens, we directly count the number of bytes in the token.

    Examples: ["bytes:\x99", "Hello", ' world', "bytes:\xe2\x80"] => 1 + 5 + 6 + 2 = 14

    The function is adapted from src/helm/proxy/static/index.js: constructTokenGroups()
    """
    num_bytes = 0
    for token in tokens:
        if token.text.startswith("bytes:"):
            num_bytes += token.text.count("\\x")
        else:
            num_bytes += len(bytes(token.text, encoding="utf-8"))
    return num_bytes


def convert_tokens_to_text(tokens: List[Token]) -> List[Dict]:
    """
    Convert tokens to strings. This function is especially useful when tokens include byte tokens.

    Example: ["<|endoftext|>", "bytes:\\xe2\\x80", "bytes:\\x99", "Hello", " world", "bytes:\\xe2\\x80",
        "bytes:\\x99", "<|endoftext|>"] => ["<|endoftext|>", "’", "Hello", " world", "’", "<|endoftext|>"]

    The function is adapted from src/helm/proxy/static/index.js: constructTokenGroups()
    """
    groups = []
    i = 0
    while i < len(tokens):
        # Aggregate consecutive tokens while they're "bytes:..."
        group: Dict = {"tokens": []}
        if tokens[i].text.startswith("bytes:"):
            bytestring = ""
            while i < len(tokens) and tokens[i].text.startswith("bytes:"):
                group["tokens"].append(tokens[i])
                # Extract part after : (e.g., \xe2\x80)
                bytestring += tokens[i].text.split(":")[1]
                i += 1
            # Convert to encoded URI (e.g., %e2%80%99) and decode
            group["text"] = unquote(bytestring.replace("\\x", "%"))
        else:
            group["tokens"].append(tokens[i])
            group["text"] = tokens[i].text
            i += 1
        groups.append(group)
    return groups


def compute_perplexity_metrics(stats: Dict[MetricName, Stat]) -> List[Stat]:
    # TODO: find out the root cause and undo num_X > 0 check
    #       https://github.com/stanford-crfm/benchmarking/issues/350
    derived_stats: List[Stat] = []

    logprob_stat = get_unique_stat_by_name(stats.values(), "logprob")
    num_tokens_stat = get_unique_stat_by_name(stats.values(), "num_perplexity_tokens")
    num_bytes_stat = get_unique_stat_by_name(stats.values(), "num_bytes")

    if logprob_stat is None:
        return []

    if num_tokens_stat is not None and num_tokens_stat.sum > 0:
        derived_stats.append(Stat(MetricName("perplexity")).add(math.e ** (-logprob_stat.sum / num_tokens_stat.sum)))

    if num_bytes_stat is not None and num_bytes_stat.sum > 0:
        derived_stats.append(
            Stat(MetricName("bits_per_byte")).add(-logprob_stat.sum / num_bytes_stat.sum / math.log(2))
        )
        derived_stats.append(Stat(MetricName("logprob_per_byte")).add(logprob_stat.sum / num_bytes_stat.sum))

    return derived_stats


class InstancesPerSplitMetric(MetricInterface):
    """Report the average num_instances in each MetricContext across train_trials."""

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        adapter_spec = scenario_state.adapter_spec
        global_stats: Dict[MetricName, Stat] = {}

        for train_trial_index in range(adapter_spec.num_train_trials):
            trial_stats: Dict[MetricName, Stat] = {}  # Statistics just for this trial
            # Group instances in this train_trial by context.
            instances_per_metric_context: Dict[MetricContext, Set[Instance]] = defaultdict(set)
            for request_state in scenario_state.request_states:
                if request_state.train_trial_index == train_trial_index:
                    instances_per_metric_context[MetricContext.from_instance(request_state.instance)].add(
                        request_state.instance
                    )
            for context, instance_set in instances_per_metric_context.items():
                stat = Stat(MetricName("num_instances")).add(len(instance_set))
                merge_stat(trial_stats, add_context(stat, context))

            # We take the mean value for each trial.
            for stat in trial_stats.values():
                merge_stat(global_stats, stat.take_mean())

        # There are no per-instance Stats.
        return MetricResult(list(global_stats.values()), [])


class BasicGenerationMetric(Metric):
    """
    Defines basic metrics which don't require domain knowledge.  This should be
    fairly comprehensive already, and we should try to use this as much as possible.
    If we need a different variant, try to generalize this or factor things out.
    It's possible we don't need to subclass this.
    `names` is a list of optional metrics to be specified by the user. Currently only `exact_match` is supported.
    """

    def __init__(self, names: List[str]):
        self.names: List[str] = names
        self.efficiency_metric = EfficiencyMetric()

    def __repr__(self):
        return f"BasicMetric({','.join(self.names)})"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Compute all metrics."""
        stats: List[Stat] = []
        stats.extend(compute_request_state_metrics(self.efficiency_metric, adapter_spec, request_state, metric_service))

        if len(request_state.instance.references) > 0:
            stats.extend(compute_reference_metrics(self.names, adapter_spec, request_state, metric_service))

        stats.extend(compute_language_modeling_metrics(adapter_spec, request_state, metric_service))

        return stats

    def derive_stats(self, stats_dict: Dict[MetricName, Stat]) -> List[Stat]:
        """Derive perplexity metrics if applicable. We don't worry about splits and perturbations here."""
        derived_stats: List[Stat] = []
        derived_stats.extend(compute_perplexity_metrics(stats_dict))
        return derived_stats

    def derive_per_instance_stats(self, per_instance_stats: Dict[Instance, List[Stat]]) -> List[Stat]:
        """Derive calibration metrics if applicable. We don't worry about splits and perturbations here."""
        derived_stats: List[Stat] = []
        derived_stats.extend(compute_calibration_metrics(per_instance_stats))
        return derived_stats


class BasicReferenceMetric(ReferenceMetric):
    """
    Defines basic metrics for Scenarios that use one Request per Reference instead of
    one per Instance.
    """

    def __init__(self):
        self.efficiency_metric = EfficiencyMetric()

    def __repr__(self):
        return "BasicReferenceMetric"

    def evaluate_references(
        self,
        adapter_spec: AdapterSpec,
        reference_request_states: List[RequestState],
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Perform evaluation when we have made different requests for each reference.
        For each reference, we have a model score (log probability) and whether it's correct.
        """

        @dataclass(frozen=True)
        class ReferenceKey:
            reference_index: int  # index of the reference
            request_mode: str  # "original" or "calibration"

        @dataclass(frozen=True)
        class ReferenceStat:
            logprob: float  # sum of logprobs for all tokens in the reference
            num_tokens: int  # number of tokens in the reference

        def compute_logprob_and_length(request_state: RequestState, window_service: WindowService) -> ReferenceStat:
            """Compute the logprob and length for the only completion from the request_state."""
            assert request_state.reference_index is not None
            assert request_state.result is not None
            assert len(request_state.result.completions) == 1

            reference_index = request_state.reference_index
            sequence: GeneratedOutput = request_state.result.completions[0]
            reference: str = request_state.instance.references[reference_index].output.text

            # Find the span of the completion that matches the reference.
            # Prepend a space because there should always be a space before reference in the prompt.
            reference_tokens: List[str] = window_service.tokenize(f" {reference}")
            num_tokens: int = len(reference_tokens)
            answer_tokens: List[Token] = sequence.tokens[-num_tokens:]
            logprob: float = sum(token.logprob for token in answer_tokens)
            assert not math.isnan(logprob), f"Log probs have NaN for RequestState: {request_state}"
            return ReferenceStat(logprob, num_tokens)

        references = reference_request_states[0].instance.references
        assert all(
            [references == request_state.instance.references for request_state in reference_request_states]
        )  # all request_state in reference_request_states should have same references
        answers = [
            reference_index for reference_index, reference in enumerate(references) if CORRECT_TAG in reference.tags
        ]
        num_choices = len(references)

        tokenizer_service: TokenizerService = metric_service
        window_service: WindowService = WindowServiceFactory.get_window_service(
            adapter_spec.model_deployment, tokenizer_service
        )
        reference_stats: Dict[ReferenceKey, ReferenceStat] = {}
        for request_state in reference_request_states:
            assert request_state.reference_index is not None and request_state.request_mode is not None
            reference_key = ReferenceKey(request_state.reference_index, request_state.request_mode)
            reference_stats[reference_key] = compute_logprob_and_length(request_state, window_service)

        if adapter_spec.method in [ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL, ADAPT_RANKING_BINARY]:
            reference_scores = [
                reference_stats[ReferenceKey(i, "original")].logprob
                / reference_stats[ReferenceKey(i, "original")].num_tokens
                for i in range(num_choices)
            ]
        elif adapter_spec.method == ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED:
            reference_scores = [
                reference_stats[ReferenceKey(i, "original")].logprob
                - reference_stats[ReferenceKey(i, "calibration")].logprob
                for i in range(num_choices)
            ]
        else:
            raise ValueError(f"Unknown adapter method: {adapter_spec.method}")

        stats: List[Stat] = []

        general_metrics: Dict[MetricName, Stat] = {}
        for request_state in reference_request_states:
            for stat in compute_request_state_metrics(
                self.efficiency_metric, adapter_spec, request_state, metric_service
            ):
                merge_stat(general_metrics, stat)
        stats.extend(general_metrics.values())
        max_prob = np.max(scipy.special.softmax(reference_scores))

        # Multiple references may attain the same maximal score; in such cases,
        # we select the first reference within the argmax list as the `predicted_index`.
        # Meanwhile, the "exact match" is calculated as the portion of correct references in the list.
        argmax_references = np.flatnonzero(reference_scores >= np.max(reference_scores))
        predicted_index = argmax_references[0].item()
        exact_match_score = len(set(answers).intersection(argmax_references)) / len(argmax_references)

        stats.extend(
            [
                Stat(MetricName("max_prob")).add(max_prob),
                Stat(MetricName("exact_match")).add(exact_match_score),
                Stat(MetricName("predicted_index")).add(predicted_index),
            ]
        )
        return stats


def compute_request_state_metrics(
    efficiency_metric: EfficiencyMetric,
    adapter_spec: AdapterSpec,
    request_state: RequestState,
    metric_service: MetricService,
) -> List[Stat]:
    """
    Compute metrics that are common to both `evaluate_generation` and `evaluate_references`.
    """
    stats: List[Stat] = []

    stats.append(Stat(MetricName("num_references")).add(len(request_state.instance.references)))

    # Copy from adapter spec
    stats.append(Stat(MetricName("num_train_trials")).add(adapter_spec.num_train_trials))

    stats.extend(efficiency_metric.compute_efficiency_metrics(adapter_spec, request_state, metric_service))
    stats.extend(_compute_finish_reason_metrics(adapter_spec, request_state, metric_service))
    stats.extend(_compute_truncation_metrics(adapter_spec, request_state, metric_service))

    return stats


def _compute_finish_reason_metrics(
    adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
) -> List[Stat]:
    """Record how often generation finished due to reaching token limit, stop token(s), or end of text"""
    assert request_state.result is not None
    sequence = request_state.result.completions[0]
    valid_reasons = [
        "length",
        "stop",
        "endoftext",
        "unknown",
    ]
    if sequence.finish_reason is None or sequence.finish_reason["reason"] not in valid_reasons:
        reason = "unknown"
    else:
        reason = sequence.finish_reason["reason"]
    return [
        Stat(MetricName(f"finish_reason_{valid_reason}")).add(int(reason == valid_reason))
        for valid_reason in valid_reasons
    ]


def _compute_truncation_metrics(
    adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
) -> List[Stat]:
    """
    Record the number of training instances used in the prompt and whether
    even the prompt needed to be truncated (once we hit zero training instances).
    """
    return [
        Stat(MetricName("num_train_instances")).add(request_state.num_train_instances),
        Stat(MetricName("prompt_truncated")).add(request_state.prompt_truncated),
    ]


def compute_language_modeling_metrics(
    adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
) -> List[Stat]:
    """Compute the logprob and normalization factors for the first completion"""
    assert request_state.result is not None
    sequence = request_state.result.completions[0]

    # Remove the empty tokens (typically generated by the AI21 tokenizer in the beginning of the text)
    #
    # Some more details about AI21 tokenizer: If the input prompt begins with a space, then
    # the tokenizer inserts an empty token to the beginning.
    # e.g. " burying him" -> ["▁"(0,0), "▁burying"(0,8), "▁him"(8,12)].
    # TODO(#1522): Update this comment once solved.
    # Since this empty token is introduced by our chunking approach, we need to remove it.
    tokens: List[Token]
    if request_state.num_conditioning_tokens > 0 and sequence.tokens[0].text == "":
        tokens = sequence.tokens[1:]
    else:
        tokens = sequence.tokens
    pred_tokens = tokens[request_state.num_conditioning_tokens :]
    logprob, num_perplexity_tokens, num_bytes = (
        sum(token.logprob for token in pred_tokens),
        len(pred_tokens),
        get_num_bytes(pred_tokens),
    )

    return [
        Stat(MetricName("logprob")).add(logprob),
        Stat(MetricName("num_perplexity_tokens")).add(num_perplexity_tokens),
        Stat(MetricName("num_bytes")).add(num_bytes),
    ]


def _has_non_zero_valued_logprobs(per_instance_stats: Dict[Instance, List[Stat]]) -> bool:
    """Return whether the per-instance stats contain non-zero-valued logprobs.

    Some models have partial functionality and produce only zero-valued logprobs."""
    for instance_stats in per_instance_stats.values():
        for stat in instance_stats:
            if stat.name.name == "logprob" and stat.sum < 0:
                return True
    return False


def compute_calibration_metrics(per_instance_stats: Dict[Instance, List[Stat]]) -> List[Stat]:
    max_probs = []
    correct = []

    # If the model does not produce non-zero-valued logprobs
    # then don't compute calibration metrics.
    if not _has_non_zero_valued_logprobs(per_instance_stats):
        hlog("Skipping computing calibration metrics because logprobs were not available.")
        return []

    for instance_stats in per_instance_stats.values():
        max_prob_stat = get_unique_stat_by_name(instance_stats, "max_prob")
        correct_stat = get_unique_stat_by_name(instance_stats, "exact_match")
        if correct_stat is not None and max_prob_stat is not None:
            assert max_prob_stat.mean is not None
            assert correct_stat.mean is not None
            max_probs.append(max_prob_stat.mean)
            cur_correct = float(correct_stat.mean)
            assert 0.0 <= cur_correct <= 1.0
            correct.append(int(cur_correct))

    stats: List[Stat] = []
    assert len(max_probs) == len(correct)
    if len(max_probs) > 0:
        # We need at least about 300 examples to compute ece_10_bin reliably.
        ece_10_bin = cal.get_ece_em(max_probs, correct, num_bins=10)
        stats.append(Stat(MetricName("ece_10_bin")).add(ece_10_bin))
        ece_1_bin = cal.get_ece(max_probs, correct, num_bins=1)
        stats.append(Stat(MetricName("ece_1_bin")).add(ece_1_bin))
        coverage_acc_area, acc_top_10_percentile = cal.get_selective_stats(max_probs, correct)
        stats.append(Stat(MetricName("selective_cov_acc_area")).add(coverage_acc_area))
        stats.append(Stat(MetricName("selective_acc@10")).add(acc_top_10_percentile))
        # Compute ECE after recalibration.
        if np.sum(correct) == 0 or np.sum(correct) == len(correct):
            # If all examples are correct or incorrect, the platt scaling
            # optimizer won't work. But our calibration error (post-calibration) will be
            # estimated as 0, so just directly store that.
            stats.append(Stat(MetricName("platt_ece_10_bin")).add(0.0))
            stats.append(Stat(MetricName("platt_ece_1_bin")).add(0.0))
        else:
            platt_scaler, clf = cal.get_platt_scaler(np.array(max_probs), np.array(correct), get_clf=True)
            stats.append(Stat(MetricName("platt_coef")).add(clf.coef_[0][0]))
            stats.append(Stat(MetricName("platt_intercept")).add(clf.intercept_[0]))
            cal_max_probs = platt_scaler(np.array(max_probs))
            platt_ece_10_bin = cal.get_ece_em(cal_max_probs, correct, num_bins=10)
            stats.append(Stat(MetricName("platt_ece_10_bin")).add(platt_ece_10_bin))
            platt_ece_1_bin = cal.get_ece(cal_max_probs, correct, num_bins=1)
            stats.append(Stat(MetricName("platt_ece_1_bin")).add(platt_ece_1_bin))

    return stats
