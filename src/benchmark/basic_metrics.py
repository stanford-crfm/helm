from dataclasses import replace
from typing import List, Callable, Dict, Optional
from urllib.parse import unquote
from functools import partial

import json
import re
import string
import nltk
from nltk.metrics.scores import f_measure
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from common.request import Token
from common.statistic import Stat
from proxy.tokenizer.tokenizer import Tokenizer
from proxy.tokenizer.tokenizer_factory import TokenizerFactory
from .augmentations.perturbation_description import PerturbationDescription
from .adapter import AdapterSpec, RequestState, ADAPT_LANGUAGE_MODELING
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService
from .tokenizer_service import TokenizerService


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")  # Required for rouge

INFERENCE_EFFICIENCY_JSON_FILEPATH = "src/benchmark/static/inference_efficiency.json"
TRAINING_EFFICIENCY_JSON_FILEPATH = "src/benchmark/static/training_efficiency.json"


def exact_match(gold: str, pred: str) -> float:
    return 1 if gold == pred else 0


def exact_match_indicator(gold: str, pred: str) -> float:
    """
    Exact match, allowing for some preceding context.
    For example, the following two answers are considered matching:
    - Because of x and y, the answer is ## <answer>
    - Given reasons y and z, the answer is ## <answer>
    While the following is considered different from the earlier two
    - Given reasons x and a, the answer is ## <other answer>
    """
    indicator: str = "#"
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
        self.tokenizer_factory: TokenizerFactory = TokenizerFactory()

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
            "exact_match_indicator": exact_match_indicator,
            "exact_set_match": exact_set_match,
            "iou_set_match": iou_set_match,
            "f1_score": f1_score,
            "rouge-1": get_rouge_function("rouge1"),
            "rouge-2": get_rouge_function("rouge2"),
            "rouge-l": get_rouge_function("rougeL"),
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

    def compute_efficiency_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute per-token normalized runtime"""
        assert request_state.result is not None

        # Compute efficiency metrics for inference.
        runtime: float = request_state.result.request_time

        # Compute total number of input and output tokens (in first sequence).
        # Fetch the right `Tokenizer` depending on the model defined in `AdapterSpec`
        # and calculate the number of tokens in the prompt.
        tokenizer_service: TokenizerService = metric_service
        tokenizer: Tokenizer = self.tokenizer_factory.get_tokenizer(adapter_spec.model, tokenizer_service)
        num_tokens_in_prompt: int = tokenizer.tokenize_and_count(request_state.request.prompt)

        sequence = request_state.result.completions[0]
        num_output_tokens: int = len(sequence.tokens[request_state.num_conditioning_tokens :])

        # The `inference_efficiency.json` file contains a `runtime_per_output_token` value
        # (the estimated runtime of generating one output token) and a
        # `runtime_for_input_tokens` dict (a mapping from various num_input_token values to
        # the estimated runtime of processing that many input tokens).
        # For example:
        # "openai/davinci": {
        #   "runtime_per_output_token": 0.08002311153903935,
        #   "runtime_for_input_tokens": {
        #     "1": 0.01592031502388136,
        #     "16": 0.01764758775115406,
        #     "32": 0.020374860478426838,
        #     ...
        #
        # These runtimes are generated by initializing Megatron with a model of the right
        # size, obtaining end-to-end generation times for different numbers of input
        # and output tokens, and then fitting a linear regression model to the
        # runtimes (slope is the runtime_per_output_token, processing time for generating
        # one token is the runtime_per_input_tokens for the corresponding num_input_tokens
        # value). Profiling code and logs, and code to fit the regression model is available
        # here: https://github.com/deepakn94/benchmarking_efficiency.
        with open(INFERENCE_EFFICIENCY_JSON_FILEPATH, "r") as f:
            inference_efficiency_dict = json.load(f)
        assert request_state.request.model in inference_efficiency_dict
        inference_efficiency_dict_for_model = inference_efficiency_dict[request_state.request.model]
        runtime_per_output_token: float = inference_efficiency_dict_for_model["runtime_per_output_token"]
        raw_runtimes_for_input_tokens: Dict[str, float] = inference_efficiency_dict_for_model[
            "runtime_for_input_tokens"
        ]
        runtimes_for_input_tokens: Dict[int, float] = {int(k): v for (k, v) in raw_runtimes_for_input_tokens.items()}
        runtime_for_input_tokens = None
        # Find the smallest num_input_tokens larger than the number of tokens in the given prompt.
        for num_input_tokens in sorted(runtimes_for_input_tokens.keys()):
            if num_tokens_in_prompt <= num_input_tokens:
                runtime_for_input_tokens = runtimes_for_input_tokens[num_input_tokens]
                break
        assert runtime_for_input_tokens is not None

        # Idealized runtime is sum of the runtime of encoding the input tokens, and the
        # runtime of generating `num_output_tokens` (`runtime_per_output_token` * (`num_output_tokens` - 1)).
        idealized_runtime: float = runtime_for_input_tokens + (runtime_per_output_token * (num_output_tokens - 1))

        # Compute efficiency metrics for training.

        # We use estimated emitted CO2 during training (in tons of CO2) as a proxy metric
        # for training efficiency. We use reported metrics where applicable, otherwise
        # we estimate them from runtime information, type and number of hardware accelerators
        # used, region, etc.
        with open(TRAINING_EFFICIENCY_JSON_FILEPATH, "r") as f:
            training_efficiency_dict = json.load(f)
        assert request_state.request.model in training_efficiency_dict
        training_co2_cost: float = training_efficiency_dict[request_state.request.model]

        return [
            Stat(MetricName("num_tokens_in_prompt")).add(num_tokens_in_prompt),
            Stat(MetricName("inference_runtime")).add(runtime),
            Stat(MetricName("inference_idealized_runtime")).add(idealized_runtime),
            Stat(MetricName("inference_runtime_discrepancy")).add(runtime - idealized_runtime),
            Stat(MetricName("training_co2_cost")).add(training_co2_cost),
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
        metrics.extend(self.compute_efficiency_metrics(adapter_spec, request_state, metric_service))

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
