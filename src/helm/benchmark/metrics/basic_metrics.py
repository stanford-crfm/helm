import math
from dataclasses import dataclass, replace
from typing import List, Callable, Optional, Dict, Tuple, Set, cast
from urllib.parse import unquote
from functools import partial

import string
import nltk
import numpy as np
import re
import scipy
import calibration as cal
from nltk.metrics.scores import f_measure
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric

from helm.common.hierarchical_logger import hlog
from helm.common.request import Token, Sequence
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
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Instance, Reference
from helm.benchmark.scenarios.math_scenario import is_equiv, is_equiv_chain_of_thought
from helm.benchmark.scenarios.code_scenario import CodeReference
from helm.benchmark.metrics.cleva_metrics_helper import ChineseTokenizer
from . import code_metrics_helper
from .metric import Metric, get_unique_stat_by_name
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")  # Required for rouge


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k).

    Numerically stable version defined in
        https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0


def quasi_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if normalize_text(gold) == normalize_text(pred) else 0


def prefix_exact_match(gold: str, pred: str) -> float:
    """
    The `prefix_exact_match` metric is particularly useful in the zero-shot setting, where the model is
    not given examples of the expected outputs and tends to output more tokens than it should.

    For example, for this zero-shot prompt from BoolQ,

    Passage: Elmendorf Air Force Base (IATA: EDF, ICAO: PAED, FAA LID: EDF) is a United States military facility
    in Anchorage, the largest city in Alaska. Originally known as Elmendorf Field, it became Elmendorf Air Force
    Base after World War II, and in 2010 it merged with nearby Fort Richardson to form Joint Base Elmendorf-Richardson.
    Question: Is there an air force base in anchorage alaska?
    Answer:

    the model could output up to `max_tokens` number of tokens "Yes, Elmendorf" instead of just "Yes".
    """
    if not pred:
        return 0

    return 1 if pred.strip().startswith(gold.strip()) else 0


def quasi_prefix_exact_match(gold: str, pred: str) -> float:
    """
    Same thing as `prefix_exact_match` but we normalize the text before checking if the prefix match.
    """
    if not pred:
        return 0

    return 1 if normalize_text(pred).startswith(normalize_text(gold)) else 0


def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
    if ret is None:  # answer is the empty string after normalizing
        return 0.0

    return ret


def exact_match_indicator(gold: str, pred: str, indicator: str = " ") -> float:
    """
    Exact match, allowing for some preceding context.
    For example, the following two answers are considered matching:
    - Because of x and y, the answer is ## <answer>
    - Given reasons y and z, the answer is ## <answer>
    While the following is considered different from the earlier two
    - Given reasons x and a, the answer is ## <other answer>
    """
    pred = pred.split(indicator)[-1].strip()
    gold = gold.split(indicator)[-1].strip()
    return exact_match(gold, pred)


def final_number_exact_match(gold: str, pred: str) -> float:
    """
    Returns 1 iff the final number in gold and pred match.
    Similar to exact_match_indicator.
    Example:
    - gold = "The answer is 15."
    - pred = "The answer is 15 eggs."
    - Returns 1
    """

    def get_final_number(x: str) -> str:
        matches = re.findall(r"-?[\d,]+(?:.\d+)?", x)
        if not matches:
            return ""
        return matches[-1].replace(",", "")

    return exact_match(get_final_number(gold), get_final_number(pred))


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


def rouge_score(gold: str, pred: str, rouge_type: str, scorer: rouge_scorer.RougeScorer) -> float:
    scores = scorer.score(gold, pred)
    return scores[rouge_type].fmeasure


def get_rouge_function(rouge_type: str) -> Callable[[str, str], float]:
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    return partial(rouge_score, scorer=scorer, rouge_type=rouge_type)


def bleu_1(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(1, 0, 0, 0))


def chinese_bleu_1(gold: str, pred: str) -> float:
    char_tokenizer = ChineseTokenizer()
    return sentence_bleu([char_tokenizer.tokenize(gold)], char_tokenizer.tokenize(pred), weights=(1, 0, 0, 0))


def get_chinese_rouge_function(rouge_type: str) -> Callable[[str, str], float]:
    char_tokenizer = ChineseTokenizer()
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True, tokenizer=char_tokenizer)
    return partial(rouge_score, scorer=scorer, rouge_type=rouge_type)


def cleva_math_result_match(gold: str, pred: str) -> float:
    """
    Exact match that only cares the last math expression.
    Common math expressions are numbers and fractions.
    """
    pattern = r"[-+*/%\.\(\)\d]+"
    matches = re.findall(pattern, pred)
    if matches:
        pred = matches[-1].lstrip(")")
    # remove space in front or at the end
    pred = pred.strip()
    return exact_match(gold, pred)


def bleu_4(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0, 0, 0, 1))


def extract_set_from_text(
    set_str: str,
    set_start_str: str = " is ",
    set_separator: str = " and ",
    empty_set_str: str = "Nothing.",
) -> Set[str]:
    """
    Given a string, extract the set of strings implied by that string.
    set_start_str denotes the start of the set
    set_separator denotes the string separating set elements
    empty_set_str is the string which denotes the empty set
    """
    if set_str == empty_set_str:
        return set()
    set_str = set_str.replace(".", "")
    extracted_set = set(set_str.split(set_start_str)[-1].split(set_separator))
    return extracted_set


def extract_gold_pred_sets(gold: str, pred: str) -> Tuple[Set[str], Set[str]]:
    """Extract the set of strings implied by the gold and pred strings"""
    gold_set = extract_set_from_text(gold)
    pred_set = extract_set_from_text(pred.split("\n")[0])
    return gold_set, pred_set


def iou_set_match(gold: str, pred: str) -> float:
    """Compute the intersection over union of the gold and pred sets"""
    gold_set, pred_set = extract_gold_pred_sets(gold, pred)
    if len(gold_set) == 0:  # If gold is empty, just check if the pred set is also empty
        return float(gold_set == pred_set)
    return len(gold_set.intersection(pred_set)) / len(gold_set.union(pred_set))


def f1_set_match(gold: str, pred: str) -> float:
    """Compute the F1 score of the gold and pred sets"""
    gold_set, pred_set = extract_gold_pred_sets(gold, pred)
    if len(gold_set) == 0:  # If gold is empty, just check if the pred set is also empty
        return float(gold_set == pred_set)
    true_positives = gold_set.intersection(pred_set)
    return 2 * len(true_positives) / (len(gold_set) + len(pred_set))


def exact_set_match(gold: str, pred: str) -> float:
    """Compute whether the sets generated exactly match"""
    gold_set, pred_set = extract_gold_pred_sets(gold, pred)
    return float(gold_set == pred_set)


def absolute_value_difference(gold: str, pred: str) -> float:
    """Compute the absolute value of the difference between two numbers (provided as strings),
    or 0.0 if invalid input.
    """

    def maybe_int(text: str):
        """Parse int, ignoring commas in numbers."""
        try:
            val = int(text.replace(",", ""))
        except ValueError:
            return 0.0
        return val

    gold_val = maybe_int(gold)
    pred_val = maybe_int(pred)
    return abs(gold_val - pred_val)


def code_eval(gold: Tuple[str, Optional[Dict]], pred: str) -> float:
    """Evaluate Code Correctness on test examples."""
    assert gold[1] is not None  # gold[1]["canonical_solution"]
    # Warning: will execute machine generated code; need to sandbox before executing
    return float(code_metrics_helper.check_correctness(gold[1], pred, 3.0)["passed"])  # type: ignore


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


class BasicMetric(Metric):
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
            name: MetricName,
            score_func: Callable,
            group: Optional[str] = None,
        ) -> List[Stat]:
            if name.name == "pass":  # Calculate pass@k for HumanEval from CodeScenario.
                score_func = cast(Callable[[Tuple[str, Optional[Dict]], str], float], score_func)  # Make mypy happy.
                code_golds = cast(List[CodeReference], golds)
                results = [
                    score_func((gold.output.text, gold.test_cases), pred) for gold in code_golds for pred in preds
                ]
                _len, _sum = len(results), int(sum(results))  # Cast to int to make type match.
                score_1 = pass_at_k_estimator(_len, _sum, 1)
                score_k = pass_at_k_estimator(_len, _sum, adapter_spec.num_outputs)
            elif name.name == "code_eval_acc":
                score_func = cast(Callable[[Tuple[str, Optional[Dict]], str], float], score_func)  # Make mypy happy.
                code_golds = cast(List[CodeReference], golds)
                score_1 = max(score_func((gold.output.text, gold.test_cases), preds[0]) for gold in code_golds)
                score_k = max(
                    score_func((gold.output.text, gold.test_cases), pred) for gold in code_golds for pred in preds
                )
            else:
                score_func = cast(Callable[[str, str], float], score_func)  # Make mypy happy.
                score_1 = max(score_func(gold.output.text, preds[0]) for gold in golds)
                score_k = max(score_func(gold.output.text, pred) for gold in golds for pred in preds)

            metrics = [Stat(name).add(score_1)]  # score_1 corresponds using one prediction
            if adapter_spec.num_outputs != 1:
                metrics.append(Stat(replace(name, name=f"{name.name}@{adapter_spec.num_outputs}")).add(score_k))
            return metrics

        # maps each string metric name to its associated function
        metric_fn_mapping: Dict[str, Callable] = {
            "exact_match": exact_match,
            "quasi_exact_match": quasi_exact_match,
            "prefix_exact_match": prefix_exact_match,
            "quasi_prefix_exact_match": quasi_prefix_exact_match,
            "exact_match_indicator": exact_match_indicator,
            "final_number_exact_match": final_number_exact_match,
            "exact_set_match": exact_set_match,
            "iou_set_match": iou_set_match,
            "f1_set_match": f1_set_match,
            "math_equiv": is_equiv,
            "math_equiv_chain_of_thought": is_equiv_chain_of_thought,
            "code_eval_acc": code_eval,
            "pass": code_eval,
            "f1_score": f1_score,
            "rouge_1": get_rouge_function("rouge1"),
            "rouge_2": get_rouge_function("rouge2"),
            "rouge_l": get_rouge_function("rougeL"),
            "bleu_1": bleu_1,
            "bleu_4": bleu_4,
            "chinese_bleu_1": chinese_bleu_1,
            "chinese_rouge_1": get_chinese_rouge_function("rouge1"),
            "chinese_rouge_2": get_chinese_rouge_function("rouge2"),
            "cleva_math_result_match": cleva_math_result_match,
            "absolute_value_difference": absolute_value_difference,
        }

        stats: List[Stat] = []

        # Gold outputs
        golds: List[Reference] = [reference for reference in request_state.instance.references if reference.is_correct]
        assert len(golds) > 0

        # Predicted outputs
        assert request_state.result is not None
        sorted_completions: List[Sequence] = sorted(request_state.result.completions, key=lambda x: -x.logprob)
        preds: List[str] = [completion.text.strip() for completion in sorted_completions]

        # Apply mapping if exists (e.g., for multiple-choice questions A -> Boston, B -> New York)
        # Note: If 'A' and 'B' were the only possible choices, smaller language models like GPT-2 would
        # sometimes predict a random letter like 'M'.
        if request_state.output_mapping is not None:
            preds = [request_state.output_mapping.get(pred) for pred in preds]  # type: ignore

        # Compute max_prob, the probability that the model assigns to its generated text.
        # Use the log prob of sorted_completions[0], which is the completion with the highest
        # log_prob. We use this since that's what's used for computing metrics like exact_match.
        # One subtlety is that when computing exact_match, we strip whitespace, so the actual
        # max_prob is the sum of all the probabilities in the set {x : strip(x) = prediction}.
        # In practice, we think this may not make much of a difference because models may not place
        # high probabilities on having additional spaces (should check this). Also, the sum
        # involves computing the log_prob for many completions which could be intractable.
        max_prob = np.exp(sorted_completions[0].logprob)
        stats.append(Stat(MetricName("max_prob")).add(max_prob))

        # Add other metrics
        for metric_name in self.names:
            if metric_name in metric_fn_mapping:
                stats.extend(compute_metrics_helper(MetricName(metric_name), metric_fn_mapping[metric_name]))
            else:
                raise NameError(f"{metric_name} is not in the list of metric functions.")

        return stats

    def compute_language_modeling_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
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
            stats.extend(self.compute_reference_metrics(adapter_spec, request_state, metric_service))

        stats.extend(self.compute_language_modeling_metrics(adapter_spec, request_state, metric_service))

        return stats

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
            sequence: Sequence = request_state.result.completions[0]
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
        for request_state in reference_request_states:
            stats.extend(
                compute_request_state_metrics(self.efficiency_metric, adapter_spec, request_state, metric_service)
            )

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

    def derive_stats(self, stats_dict: Dict[MetricName, Stat]) -> List[Stat]:
        """Derive perplexity metrics if applicable. We don't worry about splits and perturbations here."""
        derived_stats: List[Stat] = []
        derived_stats.extend(compute_perplexity_metrics(stats_dict))
        return derived_stats

    def derive_per_instance_stats(self, per_instance_stats: Dict[Instance, List[Stat]]) -> List[Stat]:
        """Derive calibration metrics if applicable. We don't worry about splits and perturbations here."""
        derived_stats: List[Stat] = []
        derived_stats.extend(compute_calibration_metrics(per_instance_stats))
        derived_stats.append(Stat(MetricName("num_instances")).add(len(per_instance_stats)))
        return derived_stats


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
