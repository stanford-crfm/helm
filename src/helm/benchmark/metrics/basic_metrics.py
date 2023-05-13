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
import importlib_resources as resources
from nltk.metrics.scores import f_measure
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

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
from . import code_metrics_helper
from .metric import Metric, get_unique_stat_by_name
from .metric_name import MetricName
from .metric_service import MetricService
from .statistic import Stat


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")  # Required for rouge


EFFICIENCY_DATA_PACKAGE: str = "helm.benchmark.efficiency_data"

INFERENCE_IDEALIZED_RUNTIMES_JSON_FILENAME: str = "inference_idealized_runtimes.json"
INFERENCE_DENOISED_RUNTIMES_JSON_FILENAME: str = "inference_denoised_runtimes.json"
TRAINING_EFFICIENCY_JSON_FILENAME: str = "training_efficiency.json"


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
        data_package = resources.files(EFFICIENCY_DATA_PACKAGE)
        with data_package.joinpath(INFERENCE_IDEALIZED_RUNTIMES_JSON_FILENAME).open("r") as f:
            self.inference_idealized_runtimes_dict = json.load(f)
        with data_package.joinpath(INFERENCE_DENOISED_RUNTIMES_JSON_FILENAME).open("r") as f:
            self.inference_denoised_runtimes_dict = json.load(f)

        # We use estimated emitted CO2 during training (in tons of CO2) as a proxy metric
        # for training efficiency. We use reported metrics where applicable, otherwise
        # we estimate them from runtime information, type and number of hardware accelerators
        # used, region, etc.
        with data_package.joinpath(TRAINING_EFFICIENCY_JSON_FILENAME).open("r") as f:
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
        assert request_state.result.request_time is not None
        runtime: float = request_state.result.request_time
        batch_size: int = 1
        # For models that perform offline batch inference, effective runtime is batch_request_time, but also
        # record batch_size to provide nuance.
        if request_state.result.batch_request_time is not None and request_state.result.batch_size is not None:
            runtime = request_state.result.batch_request_time
            batch_size = request_state.result.batch_size

        # Compute total number of prompt and output tokens.
        # Fetch the right `Tokenizer` depending on the model defined in `AdapterSpec`
        # and calculate the number of tokens in the prompt.
        tokenizer_service: TokenizerService = metric_service
        window_service: WindowService = WindowServiceFactory.get_window_service(adapter_spec.model, tokenizer_service)
        prompt: str = request_state.request.prompt
        num_prompt_tokens: int = window_service.get_num_tokens(prompt)

        # Total number of tokens in the completion.
        num_completion_tokens: int = sum([len(completion.tokens) for completion in request_state.result.completions])
        # Don't include prompt in number of generated tokens (e.g., for language modeling).
        # Assume that tokens for different completions are generated sequentially (instead of batched) when
        # computing num_output_tokens (for the purpose of runtime estimation).
        num_output_tokens: int = num_completion_tokens
        if request_state.request.echo_prompt:
            # num_prompt_tokens > num_output_tokens can happen if tokenizer doesn't round trip.
            if num_prompt_tokens <= num_output_tokens:
                num_output_tokens -= num_prompt_tokens
            else:
                hlog(
                    f"WARNING: num_prompt_tokens ({num_prompt_tokens}) > num_output_tokens ({num_output_tokens}) "
                    f"for prompt: {prompt}"
                )
                num_output_tokens = 0

        idealized_runtime: Optional[float] = compute_estimated_time_from_prompt_size_and_num_output_tokens(
            request_state, self.inference_idealized_runtimes_dict, num_prompt_tokens, num_output_tokens
        )

        denoised_runtime: Optional[float] = compute_estimated_time_from_prompt_size_and_num_output_tokens(
            request_state, self.inference_denoised_runtimes_dict, num_prompt_tokens, num_output_tokens
        )
        # Denoised runtime for offline models is just runtime.
        # We divide by batch_size to get approximate per-input runtime.
        if request_state.result.batch_size is not None:
            denoised_runtime = runtime / request_state.result.batch_size

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

        stats = [
            Stat(MetricName("num_prompt_tokens")).add(num_prompt_tokens),
            Stat(MetricName("num_completion_tokens")).add(num_completion_tokens),
            Stat(MetricName("num_output_tokens")).add(num_output_tokens),
            Stat(MetricName("inference_runtime")).add(runtime),
            Stat(MetricName("batch_size")).add(batch_size),
            Stat(MetricName("training_co2_cost")).add(training_co2_cost),
            Stat(MetricName("training_energy_cost")).add(training_energy_cost),
        ]
        if denoised_runtime is not None:
            stats.append(Stat(MetricName("inference_denoised_runtime")).add(denoised_runtime))
        if idealized_runtime is not None:
            stats.append(Stat(MetricName("inference_idealized_runtime")).add(idealized_runtime))
        return stats

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

    def compute_truncation_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """
        Record the number of training instances used in the prompt and whether
        even the prompt needed to be truncated (once we hit zero training instances).
        """
        return [
            Stat(MetricName("num_train_instances")).add(request_state.num_train_instances),
            Stat(MetricName("prompt_truncated")).add(request_state.prompt_truncated),
        ]

    def compute_all_general_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """
        Compute metrics that are common to both `evaluate_generation` and `evaluate_references`.
        """
        stats: List[Stat] = []

        stats.append(Stat(MetricName("num_references")).add(len(request_state.instance.references)))

        # Copy from adapter spec
        stats.append(Stat(MetricName("num_train_trials")).add(adapter_spec.num_train_trials))

        stats.extend(self.compute_efficiency_metrics(adapter_spec, request_state, metric_service))
        stats.extend(self.compute_finish_reason_metrics(adapter_spec, request_state, metric_service))
        stats.extend(self.compute_truncation_metrics(adapter_spec, request_state, metric_service))

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
        stats.extend(self.compute_all_general_metrics(adapter_spec, request_state, metric_service))

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
        window_service: WindowService = WindowServiceFactory.get_window_service(adapter_spec.model, tokenizer_service)
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
        stats.extend(self.compute_all_general_metrics(adapter_spec, request_state, metric_service))

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


def compute_calibration_metrics(per_instance_stats: Dict[Instance, List[Stat]]) -> List[Stat]:
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
