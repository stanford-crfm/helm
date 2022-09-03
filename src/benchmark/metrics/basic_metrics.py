import math
from dataclasses import dataclass, replace
from typing import List, Callable, Optional, Dict, Tuple, Set, cast
from urllib.parse import unquote
from functools import partial

import json
import re
import string
import nltk
import numpy as np
import scipy
import calibration as cal
from nltk.metrics.scores import f_measure
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from common.request import Token, Sequence
from common.general import singleton
from benchmark.adapter import (
    ADAPT_MULTIPLE_CHOICE_JOINT,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED,
    AdapterSpec,
    RequestState,
)
from benchmark.window_services.window_service import WindowService
from benchmark.window_services.window_service_factory import WindowServiceFactory
from benchmark.window_services.tokenizer_service import TokenizerService
from benchmark.scenarios.scenario import CORRECT_TAG, Instance
from benchmark.scenarios.math_scenario import is_equiv, is_equiv_chain_of_thought
from benchmark.scenarios.code_scenario import CodeReference
from . import code_metrics_helper
from .metric import Metric, get_unique_stat_by_name
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")  # Required for rouge

INFERENCE_IDEALIZED_RUNTIMES_JSON_FILEPATH: str = "src/benchmark/static/inference_idealized_runtimes.json"
INFERENCE_DENOISED_RUNTIMES_JSON_FILEPATH: str = "src/benchmark/static/inference_denoised_runtimes.json"
TRAINING_EFFICIENCY_JSON_FILEPATH: str = "src/benchmark/static/training_efficiency.json"


def compute_estimated_time_from_prompt_size_and_num_output_tokens(
    request_state: RequestState,
    inference_runtimes_dict: Dict[str, Dict],
    num_prompt_tokens: int,
    num_output_tokens: int,
) -> Optional[float]:
    estimated_runtime: Optional[float]
    if request_state.request.model in inference_runtimes_dict:
        inference_runtimes_dict_for_model = inference_runtimes_dict[request_state.request.model]
        runtime_per_output_token: float = inference_runtimes_dict_for_model["runtime_per_output_token"]
        raw_runtimes_for_prompt_tokens: Dict[str, float] = inference_runtimes_dict_for_model[
            "runtime_for_prompt_tokens"
        ]
        runtimes_for_prompt_tokens: Dict[int, float] = {int(k): v for (k, v) in raw_runtimes_for_prompt_tokens.items()}

        runtime_for_prompt_tokens: Optional[float] = None
        largest_num_tokens_in_efficiency_dict: int = max(runtimes_for_prompt_tokens.keys())
        # Find the smallest num_prompt_tokens larger than the number of tokens in the given prompt,
        # then scale runtime in dict by (num_prompt_tokens / key) to get more accurate estimate: we
        # assume that we can encode the prompt at the same throughput as the smallest key larger than
        # num_prompt_tokens, and number of compute operations scales linearly with num_prompt_tokens.
        for key in sorted(runtimes_for_prompt_tokens.keys()):
            if num_prompt_tokens <= key:
                runtime_for_prompt_tokens = runtimes_for_prompt_tokens[key] * (num_prompt_tokens / key)
                break
        # If number of tokens in the prompt exceeds the largest key in the efficiency dict, then
        # estimate the prompt encoding time by linearly scaling up the runtime for the largest
        # key (this is reasonably accurate under certain simplifying assumptions).
        if runtime_for_prompt_tokens is None:
            runtime_for_prompt_tokens = runtimes_for_prompt_tokens[largest_num_tokens_in_efficiency_dict] * (
                num_prompt_tokens / largest_num_tokens_in_efficiency_dict
            )
        overhead: Optional[float] = inference_runtimes_dict_for_model.get("overhead")

        # Idealized runtime is sum of the runtime of encoding the input tokens, the runtime of
        # generating `num_output_tokens` (`runtime_per_output_token` * (`num_output_tokens` - 1))
        # if number of output tokens is greater than 0, otherwise just `runtime_for_prompt_tokens`,
        # and the overhead if available.
        estimated_runtime = runtime_for_prompt_tokens
        if num_output_tokens > 0:
            estimated_runtime += runtime_per_output_token * (num_output_tokens - 1)
        # Add overhead if it is available.
        if overhead is not None:
            estimated_runtime += overhead
    else:
        estimated_runtime = None

    return estimated_runtime


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


def get_num_bytes(tokens: List[Token]) -> int:
    """
    Compute the byte length of the input tokens. For a UTF-8 string token, we use byte() to convert
    it to bytes; for byte tokens, we directly count the number of bytes in the token.

    Examples: ["bytes:\x99", "Hello", ' world', "bytes:\xe2\x80"] => 1 + 5 + 6 + 2 = 14

    The function is adapted from src/proxy/static/index.js: constructTokenGroups()
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

    The function is adapted from src/proxy/static/index.js: constructTokenGroups()
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


def bleu_4(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0, 0, 0, 1))


def extract_set_from_text(
    set_str: str, set_start_str: str = " is ", set_separator: str = " and ", empty_set_str: str = "Nothing.",
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

        # For Efficiency metrics:
        # The `inference_efficiency.json` file contains a `runtime_per_output_token` value
        # (the estimated runtime of generating one output token) and a
        # `runtime_for_prompt_tokens` dict (a mapping from various num_prompt_tokens values to
        # the estimated runtime of encoding a prompt with that many tokens).
        # For example:
        # "openai/davinci": {
        #   "runtime_per_output_token": 0.080,
        #   "runtime_for_prompt_tokens": {
        #     "1": 0.016,
        #     "16": 0.018,
        #     "32": 0.020,
        #     ...
        #
        # These runtimes are generated by initializing Megatron with a model of the right size,
        # obtaining end-to-end generation times for different numbers of prompt and output tokens,
        # and then fitting a linear regression model to the runtimes: the resulting slope is the
        # runtime_per_output_token, which is the processing time for generating each output token,
        # and the y-intercept is the runtime_for_prompt_tokens, with different values for different
        # num_prompt_tokens values.
        # Profiling code and logs, and code to fit the regression model is available at
        # https://github.com/stanford-crfm/benchmarking_efficiency.
        with open(INFERENCE_IDEALIZED_RUNTIMES_JSON_FILEPATH, "r") as f:
            self.inference_idealized_runtimes_dict = json.load(f)
        with open(INFERENCE_DENOISED_RUNTIMES_JSON_FILEPATH, "r") as f:
            self.inference_denoised_runtimes_dict = json.load(f)

        # We use estimated emitted CO2 during training (in tons of CO2) as a proxy metric
        # for training efficiency. We use reported metrics where applicable, otherwise
        # we estimate them from runtime information, type and number of hardware accelerators
        # used, region, etc.
        with open(TRAINING_EFFICIENCY_JSON_FILEPATH, "r") as f:
            self.training_efficiency_dict = json.load(f)

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

        def compute_metrics_helper(name: MetricName, score_func: Callable, group: Optional[str] = None,) -> List[Stat]:
            if name.name == "pass":  # Calculate pass@k for HumanEval from CodeScenario.
                score_func = cast(Callable[[Tuple[str, Optional[Dict]], str], float], score_func)  # Make mypy happy.
                code_golds = cast(List[CodeReference], golds)
                results = [score_func((gold.output, gold.test_cases), pred) for gold in code_golds for pred in preds]
                _len, _sum = len(results), int(sum(results))  # Cast to int to make type match.
                score_1 = pass_at_k_estimator(_len, _sum, 1)
                score_k = pass_at_k_estimator(_len, _sum, adapter_spec.num_outputs)
            elif name.name == "code_eval_acc":
                score_func = cast(Callable[[Tuple[str, Optional[Dict]], str], float], score_func)  # Make mypy happy.
                code_golds = cast(List[CodeReference], golds)
                score_1 = max(score_func((gold.output, gold.test_cases), preds[0]) for gold in code_golds)
                score_k = max(score_func((gold.output, gold.test_cases), pred) for gold in code_golds for pred in preds)
            else:
                score_func = cast(Callable[[str, str], float], score_func)  # Make mypy happy.
                score_1 = max(score_func(gold.output, preds[0]) for gold in golds)
                score_k = max(score_func(gold.output, pred) for gold in golds for pred in preds)

            metrics = [Stat(name).add(score_1)]  # score_1 corresponds using one prediction
            if adapter_spec.num_outputs != 1:
                metrics.append(Stat(replace(name, name=f"{name.name}@{adapter_spec.num_outputs}")).add(score_k))
            return metrics

        # maps each string metric name to its associated function
        metric_fn_mapping: Dict[str, Callable] = {
            "exact_match": exact_match,
            "quasi_exact_match": quasi_exact_match,
            "exact_match_indicator": exact_match_indicator,
            "exact_set_match": exact_set_match,
            "iou_set_match": iou_set_match,
            "f1_set_match": f1_set_match,
            "math_equiv": is_equiv,
            "math_equiv_chain_of_thought": is_equiv_chain_of_thought,
            "code_eval_acc": code_eval,
            "pass": code_eval,
            "f1_score": f1_score,
            "rouge-1": get_rouge_function("rouge1"),
            "rouge-2": get_rouge_function("rouge2"),
            "rouge-l": get_rouge_function("rougeL"),
            "bleu_1": bleu_1,
            "bleu_4": bleu_4,
            "absolute_value_difference": absolute_value_difference,
        }

        reference_metrics = []
        # Gold outputs
        golds = [reference for reference in request_state.instance.references if reference.is_correct]
        assert len(golds) > 0

        # Predicted outputs
        assert request_state.result is not None
        # TODO: Sort the predictions, or take them from the top tokens of the first completion
        #       https://github.com/stanford-crfm/benchmarking/issues/42
        preds = [completion.text.strip() for completion in request_state.result.completions]

        # Apply mapping if exists (e.g., for multiple-choice questions A -> Boston, B -> New York)
        # Note: If 'A' and 'B' were the only possible choices, smaller language models like GPT-2 would
        # sometimes predict a random letter like 'M'.
        if request_state.output_mapping is not None:
            preds = [request_state.output_mapping.get(pred) for pred in preds]

        # Add calibration metrics for ADAPT_MULTIPLE_CHOICE_JOINT.
        if adapter_spec.method == ADAPT_MULTIPLE_CHOICE_JOINT:
            max_prob = np.exp(singleton(request_state.result.completions).logprob)
            reference_metrics.append(Stat(MetricName("max_prob")).add(max_prob))

        # Add other metrics.
        for metric_name in self.names:
            if metric_name in metric_fn_mapping:
                reference_metrics.extend(
                    compute_metrics_helper(MetricName(metric_name), metric_fn_mapping[metric_name])
                )
            else:
                raise NameError(f"{metric_name} is not in the list of metric functions.")
        return reference_metrics

    def compute_efficiency_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute efficiency metrics for both inference and training.
        For inference, we record both the actual runtime and an estimated idealized runtime
        for the given request with an optimized software implementation run on A100 GPU(s),
        taking into account both the number of tokens in the prompt of the request, and the
        number of generated output tokens.
        For training, we report the estimated total metric tons of CO2 emitted to train the
        model. This is the same for each request."""
        assert request_state.result is not None
        # Compute efficiency metrics for inference.
        runtime: float = request_state.result.request_time
        batch_size: int = 1
        # For models that perform offline batch inference, effective runtime is batch_request_time, but also
        # record batch_size to provide nuance.
        if request_state.result.batch_request_time is not None and request_state.result.batch_size is not None:
            runtime = request_state.result.batch_request_time
            batch_size = request_state.result.batch_size

        # Compute total number of prompt and output tokens (in first sequence).
        # Fetch the right `Tokenizer` depending on the model defined in `AdapterSpec`
        # and calculate the number of tokens in the prompt.
        tokenizer_service: TokenizerService = metric_service
        window_service: WindowService = WindowServiceFactory.get_window_service(adapter_spec.model, tokenizer_service)
        prompt: str = request_state.request.prompt
        num_prompt_tokens: int = window_service.get_num_tokens(prompt)

        sequence = request_state.result.completions[0]
        num_output_tokens: int = len(sequence.tokens)
        # Don't include prompt in number of generated tokens (e.g., for language modeling).
        if request_state.request.echo_prompt:
            # This might fail when we get fewer output tokens in the response than the number of tokens in the prompt.
            assert (
                num_prompt_tokens <= num_output_tokens
            ), f"num_prompt_tokens ({num_prompt_tokens}) > num_output_tokens ({num_output_tokens}) for prompt: {prompt}"
            num_output_tokens -= num_prompt_tokens

        idealized_runtime: Optional[float] = compute_estimated_time_from_prompt_size_and_num_output_tokens(
            request_state, self.inference_idealized_runtimes_dict, num_prompt_tokens, num_output_tokens
        )

        denoised_runtime: Optional[float] = compute_estimated_time_from_prompt_size_and_num_output_tokens(
            request_state, self.inference_denoised_runtimes_dict, num_prompt_tokens, num_output_tokens
        )

        # Compute efficiency metrics for training.
        training_co2_cost: Optional[float]
        if request_state.request.model in self.training_efficiency_dict["carbon"]:
            training_co2_cost = self.training_efficiency_dict["carbon"][request_state.request.model]["value"]
        else:
            training_co2_cost = None

        training_energy_cost: Optional[float]
        if request_state.request.model in self.training_efficiency_dict["energy"]:
            training_energy_cost = self.training_efficiency_dict["energy"][request_state.request.model]["value"]
        else:
            training_energy_cost = None

        return [
            Stat(MetricName("num_prompt_tokens")).add(num_prompt_tokens),
            Stat(MetricName("num_output_tokens")).add(num_output_tokens),
            Stat(MetricName("inference_runtime")).add(runtime),
            Stat(MetricName("batch_size")).add(batch_size),
            Stat(MetricName("inference_denoised_runtime")).add(denoised_runtime),
            Stat(MetricName("inference_idealized_runtime")).add(idealized_runtime),
            Stat(MetricName("training_co2_cost")).add(training_co2_cost),
            Stat(MetricName("training_energy_cost")).add(training_energy_cost),
        ]

    def compute_finish_reason_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
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

    def compute_num_in_context_examples(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Record the number of in-context examples used in the prompt."""
        return [Stat(MetricName("num_in_context_examples")).add(request_state.num_in_context_examples)]

    def compute_input_truncated(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Record whether the input was truncated to fit the context window."""
        return [Stat(MetricName("input_truncated")).add(request_state.input_truncated)]

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
        """Compute the reference metrics and language modeling metrics"""
        metrics = []
        if len(request_state.instance.references) > 0:
            metrics.extend(self.compute_reference_metrics(adapter_spec, request_state, metric_service))

        metrics.extend(self.compute_language_modeling_metrics(adapter_spec, request_state, metric_service))
        metrics.extend(self.compute_efficiency_metrics(adapter_spec, request_state, metric_service))
        metrics.extend(self.compute_finish_reason_metrics(adapter_spec, request_state, metric_service))
        metrics.extend(self.compute_num_in_context_examples(adapter_spec, request_state, metric_service))
        metrics.extend(self.compute_input_truncated(adapter_spec, request_state, metric_service))

        return metrics

    def evaluate_references(
        self,
        adapter_spec: AdapterSpec,
        reference_request_states: List[RequestState],
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Setup: for each reference, we have a model score (log probability) and whether it's correct.
        We define the following metrics:
        - correct_rank: if we sort references by their logprobs, what is the ranking of the first correct reference.
        """
        # TODO: https://github.com/stanford-crfm/benchmarking/issues/45

        @dataclass(frozen=True)
        class ReferenceKey:
            reference_index: int  # index of the reference
            request_mode: str  # "original" or "calibration"

        @dataclass(frozen=True)
        class ReferenceStat:
            logprob: float  # sum of logprobs for all tokens in the reference
            num_tokens: int  # number of tokens in the reference

        def alphanumeric_filter(text: str):
            alphanumeric_chars: str = string.digits + string.ascii_lowercase + string.ascii_uppercase
            return "".join(list(filter(lambda x: x in alphanumeric_chars, text)))

        def compute_logprob_and_length(request_state: RequestState) -> ReferenceStat:
            """Compute the logprob and length for the only completion from the request_state."""
            assert request_state.reference_index is not None
            assert request_state.result is not None
            assert len(request_state.result.completions) == 1

            reference_index = request_state.reference_index
            sequence: Sequence = request_state.result.completions[0]
            reference: str = request_state.instance.references[reference_index].output

            # Find the span of the completion that matches the reference.
            answer_tokens: List[Token] = []
            for token in sequence.tokens[::-1]:
                span: str = "".join([token.text for token in answer_tokens])
                if alphanumeric_filter(span) == alphanumeric_filter(reference):
                    break
                answer_tokens.insert(0, token)

            # Sanity check
            span: str = "".join([token.text for token in answer_tokens])
            assert alphanumeric_filter(span) == alphanumeric_filter(reference), f"Expected: {reference}, Actual: {span}"

            logprob = sum(token.logprob for token in answer_tokens)
            num_tokens = len(answer_tokens)

            return ReferenceStat(logprob, num_tokens)

        references = reference_request_states[0].instance.references
        assert all(
            [references == request_state.instance.references for request_state in reference_request_states]
        )  # all request_state in reference_request_states should have same references
        answers = [
            reference_index for reference_index, reference in enumerate(references) if CORRECT_TAG in reference.tags
        ]
        num_choices = len(references)

        reference_stats: Dict[ReferenceKey, ReferenceStat] = {}
        for request_state in reference_request_states:
            assert request_state.reference_index is not None and request_state.request_mode is not None
            reference_key = ReferenceKey(request_state.reference_index, request_state.request_mode)
            reference_stats[reference_key] = compute_logprob_and_length(request_state)

        if adapter_spec.method == ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL:
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

        answer_scores = [reference_scores[i] for i in answers]

        max_prob = np.max(scipy.special.softmax(reference_scores))
        # TODO: fix the accuracy calculation---it is currently incorrect when there are ties.
        # For example, in binary classification if the model's predicted probabilities are
        # [0.5, 0.5], then this code will say we got the example correct, regardless of what
        # the answer is, because the calculation max(reference_scores) == max(answer_scores)
        # will always return True.
        return [
            Stat(MetricName("max_prob")).add(max_prob),
            Stat(MetricName("exact_match")).add(float(max(reference_scores) == max(answer_scores))),
        ]

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


def compute_calibration_metrics(per_instance_stats: Dict[Instance, List[Stat]]):
    max_probs = []
    correct = []
    for instance_stats in per_instance_stats.values():
        max_prob_stat = get_unique_stat_by_name(instance_stats, "max_prob")
        correct_stat = get_unique_stat_by_name(instance_stats, "exact_match")
        if correct_stat is not None and max_prob_stat is not None:
            assert max_prob_stat.mean is not None
            assert correct_stat.mean is not None
            max_probs.append(max_prob_stat.mean)
            cur_correct = float(correct_stat.mean)
            # For a single example, we either get it correct or not.
            assert np.isclose(cur_correct, 1.0) or np.isclose(cur_correct, 0.0)
            correct.append(int(cur_correct))

    calibration_metrics: List[Stat] = []
    assert len(max_probs) == len(correct)
    if len(max_probs) > 0:
        # We need at least around 300 examples to compute ece_10_bin reliably.
        ece_10_bin = cal.get_ece_em(max_probs, correct, num_bins=10)
        calibration_metrics.append(Stat(MetricName("ece_10_bin")).add(ece_10_bin))
        ece_1_bin = cal.get_ece(max_probs, correct, num_bins=1)
        calibration_metrics.append(Stat(MetricName("ece_1_bin")).add(ece_1_bin))
        coverage_acc_area, acc_top_10_percentile = cal.get_selective_stats(max_probs, correct)
        calibration_metrics.append(Stat(MetricName("selective_cov_acc_area")).add(coverage_acc_area))
        calibration_metrics.append(Stat(MetricName("selective_acc@10")).add(acc_top_10_percentile))
    return calibration_metrics
