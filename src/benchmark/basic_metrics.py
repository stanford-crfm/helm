from dataclasses import replace
from typing import List, Callable, Dict, Optional
from urllib.parse import unquote

import re
import string
import rouge
import nltk
from nltk.metrics.scores import f_measure
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from common.request import Token
from common.statistic import Stat
from proxy.tokenizer.auto_token_counter import AutoTokenCounter
from proxy.tokenizer.token_counter import TokenCounter
from .augmentations.perturbation_description import PerturbationDescription
from .adapter import AdapterSpec, RequestState, ADAPT_LANGUAGE_MODELING
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")  # Required for rouge


def exact_match(gold: str, pred: str) -> float:
    return 1 if gold == pred else 0


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


# TODO should we be normalizing everything this way? (e.g., iou_set_match)
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


def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
    if ret is None:  # answer is the empty string after normalizing
        return 0.0

    return ret


def rouge_l(gold: str, pred: str) -> float:
    rouge_l_evaluator = rouge.Rouge(
        metrics=["rouge-l"], weight_factor=1.2,  # Original Rouge Paper uses 1.2, https://aclanthology.org/W04-1013.pdf
    )
    score: dict = rouge_l_evaluator.get_scores(pred, gold)
    return score["rouge-l"]["f"]


def bleu_1(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(1, 0, 0, 0))


def bleu_4(gold: str, pred: str) -> float:
    return sentence_bleu([word_tokenize(gold)], word_tokenize(pred), weights=(0, 0, 0, 1))


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

    def __init__(self, names: List[str]):
        self.names: List[str] = names
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

        def compute_metrics_helper(name: MetricName, score_func: Callable[[str, str], float]) -> List[Stat]:
            score_1 = max(score_func(gold, preds[0]) for gold in golds)
            score_k = max(score_func(gold, pred) for gold in golds for pred in preds)
            return [
                Stat(name).add(score_1),
                Stat(replace(name, k=adapter_spec.num_outputs)).add(score_k),
            ]

        # maps each string metric name to its associated function
        metric_fn_mapping = {
            "exact_match": exact_match,
            "exact_set_match": exact_set_match,
            "iou_set_match": iou_set_match,
            "f1_score": f1_score,
            "rouge-l": rouge_l,
            "bleu_1": bleu_1,
            "bleu_4": bleu_4,
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
                preds = [completion.text.strip() for completion in request_state.result.completions]

                # Apply mapping if exists (e.g., for multiple-choice questions A -> Boston, B -> New York)
                if request_state.output_mapping is not None:
                    preds = [request_state.output_mapping.get(pred) for pred in preds]
                reference_metrics.extend(
                    compute_metrics_helper(MetricName(metric_name), metric_fn_mapping[metric_name])
                )

                perturbation: Optional[PerturbationDescription] = request_state.instance.perturbation
                if perturbation:
                    reference_metrics.extend(
                        compute_metrics_helper(
                            MetricName(metric_name, perturbation=perturbation), metric_fn_mapping[metric_name]
                        )
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

        # Compute total number of input and output tokens
        num_tokens_in_prompt: int = self.token_counter.tokenize_and_count(
            model=request_state.request.model, text=request_state.request.prompt
        )
        num_tokens: int = sum([len(sequence.tokens) for sequence in request_state.result.completions])
        num_output_tokens: int = num_tokens
        if request_state.request.echo_prompt:  # Subtract out num_tokens_in_prompt
            num_output_tokens -= num_tokens_in_prompt * len(request_state.result.completions)

        runtime_per_output_token = {
            "openai/ada": 0.020,
            "openai/davinci": 0.064,
            "ai21/j1_large": 0.026,
            "ai21/j1_jumbo": 0.056,
        }
        # TODO: This is just the measured runtime for 256 input tokens. Do
        # something smarter.
        runtime_for_all_input_tokens = {
            "openai/ada": 0.016,
            "openai/davinci": 0.302,
            "ai21/j1_large": 0.040,
            "ai21/j1_jumbo": 0.229,
        }
        estimated_runtime: float = runtime_for_all_input_tokens[request_state.request.model] + (
            runtime_per_output_token[request_state.request.model] * num_output_tokens
        )

        return [
            Stat(MetricName("runtime")).add(runtime),
            Stat(MetricName("estimated_runtime")).add(estimated_runtime),
            Stat(MetricName("runtime_overhead")).add(runtime - estimated_runtime),
        ]

    def compute_language_modeling_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the logprob and normalization factors for the first completion"""
        assert request_state.result is not None
        sequence = request_state.result.completions[0]

        # For LM, the prompt and the response should equal
        if adapter_spec.method == ADAPT_LANGUAGE_MODELING:
            assert (
                "".join([group["text"] for group in convert_tokens_to_text(sequence.tokens)])
                == request_state.request.prompt
            )

        pred_tokens = sequence.tokens[request_state.num_conditioning_tokens :]
        logprob, num_tokens, num_bytes = (
            sum(token.logprob for token in pred_tokens),
            len(pred_tokens),
            get_num_bytes(pred_tokens),
        )

        return [
            Stat(MetricName("logprob")).add(logprob),
            Stat(MetricName("num_tokens")).add(num_tokens),
            Stat(MetricName("num_bytes")).add(num_bytes),
        ]

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
