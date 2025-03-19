from dataclasses import replace
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple, cast
import re
import string

from nltk.metrics.scores import f_measure
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics import code_metrics_helper
from helm.benchmark.metrics.cleva_metrics_helper import ChineseTokenizer
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.nltk_helper import install_nltk_resources
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.code_scenario import CodeReference
from helm.benchmark.scenarios.math_scenario import is_equiv, is_equiv_chain_of_thought
from helm.benchmark.scenarios.scenario import Reference
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import GeneratedOutput


install_nltk_resources()


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k).

    Numerically stable version defined in
        https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def normalize_text(text: str, should_remove_articles: bool = True) -> str:
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

    normalized_text = remove_punc(lower(text))
    if should_remove_articles:
        normalized_text = remove_articles(normalized_text)
    return white_space_fix(normalized_text)


def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0


def quasi_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if normalize_text(gold) == normalize_text(pred) else 0


def quasi_leave_articles_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return (
        1
        if normalize_text(gold, should_remove_articles=False) == normalize_text(pred, should_remove_articles=False)
        else 0
    )


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


def cider(gold: str, pred: str) -> float:
    try:
        from pycocoevalcap.cider.cider import Cider
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["vlm"])

    cider_evaluator = Cider()
    candidate = {"caption": [pred]}
    reference = {"caption": [gold]}
    average_score, _ = cider_evaluator.compute_score(reference, candidate)
    return average_score


def wer_score(gold: str, pred: str) -> float:
    # Word Error Rate (WER), which is a common
    # metric used to evaluate the accuracy of speech recognition systems.
    # The lower the better. The WER might be greater than 1.
    # https://huggingface.co/learn/audio-course/en/chapter5/evaluation#word-error-rate
    try:
        from jiwer import wer
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    if not pred:
        return 0
    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    wer_ret = wer(gold, pred)
    return wer_ret


def mer_score(gold: str, pred: str) -> float:
    # Match Error Rate (MER), which is for evaluating the error rate of
    # speech recognition systems. The lower the better.
    try:
        from jiwer import mer
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    if not pred:
        return 0

    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    mer_ret = mer(gold, pred)
    return mer_ret


def wip_score(gold: str, pred: str) -> float:
    # Word information preservation (WIP) for evaluating the preserved information of speech
    # recognition systems. The higher the better.
    try:
        from jiwer import wip
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    if not pred:
        return 0

    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    wip_ret = wip(gold, pred)
    return wip_ret


def cer_score(gold: str, pred: str) -> float:
    # Character Error Rate (CER) for evaluating the accuracy
    # of speech recognition systems. The lower the better.
    try:
        from jiwer import cer
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    if not pred:
        return 0

    gold = normalize_text(gold, should_remove_articles=False)
    pred = normalize_text(pred, should_remove_articles=False)
    cer_ret = cer(gold, pred)
    assert isinstance(cer_ret, float)
    return cer_ret


def chinese_wer_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    return wer_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


def chinese_mer_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    return mer_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


def chinese_wip_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    return wip_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


def chinese_cer_score(gold: str, pred: str) -> float:
    try:
        import jieba
    except ModuleNotFoundError as e:
        handle_module_not_found_error(e, ["audiolm"])

    return cer_score(" ".join(jieba.cut(gold)), " ".join(jieba.cut(pred)))


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


# TODO This should probably be made into an implementation of MetricInterface. For now it lives here
# just to separate it from basic_metrics.py.
def compute_reference_metrics(
    names: List[str], adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
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
            results = [score_func((gold.output.text, gold.test_cases), pred) for gold in code_golds for pred in preds]
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
        "quasi_leave_articles_exact_match": quasi_leave_articles_exact_match,
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
        "cider": cider,
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
        "wer_score": wer_score,
        "mer_score": mer_score,
        "wip_score": wip_score,
        "cer_score": cer_score,
        "chinese_wer_score": chinese_wer_score,
        "chinese_mer_score": chinese_mer_score,
        "chinese_wip_score": chinese_wip_score,
        "chinese_cer_score": chinese_cer_score,
    }

    stats: List[Stat] = []

    # Gold outputs
    golds: List[Reference] = [reference for reference in request_state.instance.references if reference.is_correct]
    assert len(golds) > 0

    # Predicted outputs
    assert request_state.result is not None
    sorted_completions: List[GeneratedOutput] = sorted(request_state.result.completions, key=lambda x: -x.logprob)
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
    for metric_name in names:
        if metric_name in metric_fn_mapping:
            stats.extend(compute_metrics_helper(MetricName(metric_name), metric_fn_mapping[metric_name]))
        else:
            raise NameError(f"{metric_name} is not in the list of metric functions.")

    return stats
