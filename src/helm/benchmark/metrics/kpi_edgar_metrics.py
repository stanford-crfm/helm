from typing import List, Dict, Set, Tuple, Callable, Union, cast
import logging
import re
import itertools
import statistics

from helm.common.request import Sequence
from helm.benchmark.adaptation.request_state import RequestState
from .metric import Metric
from .metric_name import MetricName
from .statistic import Stat
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .metric_service import MetricService
from helm.benchmark.scenarios.scenario import Reference
from helm.benchmark.scenarios.kpi_edgar_scenario import TAG_DICT, TAG_PAREN_RE

DEFAULT_TAG_PAREN_RE = (r"\(", r"\)")


def tokenize(text: str) -> List[str]:
    # TODO: Better to introduce a sophisticated tokenizer to support (multilingal) natural texts.
    return text.strip().split(" ")


def get_tagged_token_dict(token_list: List[str]) -> Dict[str, Set[Tuple[int, str]]]:
    # TODO: Note: We need to handle the cases where the original text contains < or > to avoid the confusion with tags.
    # TODO: Maybe better to introduce XML parser or more sophisticated parser.

    tagged_token_dict: Dict[str, Set[Tuple[int, str]]] = {tag: set() for tag in TAG_DICT.keys()}
    curr_tag = "O"
    for (idx, token) in enumerate(token_list):
        sub_token_list = re.split("[<>]", token)
        logging.debug(sub_token_list)
        curr_token = sub_token_list[0]
        if token.startswith("<"):
            tag = sub_token_list[1]
            curr_tag = tag if tag in tagged_token_dict.keys() else curr_tag
            curr_token = sub_token_list[2]
        if curr_tag != "O":
            tagged_token_dict[curr_tag] = tagged_token_dict[curr_tag].union({(idx, curr_token)})
        if token.endswith(">") and sub_token_list[-2].startswith("/"):
            tag = sub_token_list[-1][1:]
            curr_tag = "O"
    return tagged_token_dict


def get_tagged_token_size_dict(token_list: List[str]) -> Tuple[Dict[str, Set], Dict[str, int]]:

    tagged_token_dict = get_tagged_token_dict(token_list)
    tagged_size_dict: Dict[str, int] = {tag: len(st) for (tag, st) in tagged_token_dict.items()}
    return (tagged_token_dict, tagged_size_dict)


def get_intersection(
    gold_set: Set[Tuple[int, str]], pred_set: Set[Tuple[int, str]], ignore_index: bool
) -> Set[Tuple[int, str]]:
    def remove_index(the_set: Set[Tuple[int, str]]) -> Set[Tuple[int, str]]:
        return {(0, e[1]) for e in the_set}

    tmp_gold_set = remove_index(gold_set) if ignore_index else gold_set
    tmp_pred_set = remove_index(pred_set) if ignore_index else pred_set

    return tmp_gold_set.intersection(tmp_pred_set)


def get_tag_and_phrase(extracted: str, re_tag_paren: Tuple[str, str] = DEFAULT_TAG_PAREN_RE) -> Tuple[str, str]:
    matched = re.match(r"(.*)%s(.*)%s" % (re_tag_paren[0], re_tag_paren[1]), extracted)
    sub_token_tpl = matched.groups() if matched is not None else tuple()
    if len(sub_token_tpl) == 2:
        phrase = sub_token_tpl[0].strip()
        tag = sub_token_tpl[1].strip()
        return (tag, phrase)
    return ("", "")


def get_tagged_token_size_dict_extraction(
    entity_list: List[str], re_tag_paren: Tuple[str, str] = DEFAULT_TAG_PAREN_RE
) -> Tuple[Dict[str, Set], Dict[str, int]]:

    tmp_tag_and_phrase_list = [get_tag_and_phrase(entity, re_tag_paren) for entity in entity_list]
    tag_and_phrase_list = [tp for tp in tmp_tag_and_phrase_list if len(tp[0]) != 0]
    tagged_token_dict: Dict[str, Set[Tuple[int, str]]] = {tag: set() for tag in TAG_DICT.keys()}
    for (tag, phrase) in tag_and_phrase_list:
        if tag in tagged_token_dict.keys():
            word_list = phrase.split(" ")
            token_list = [(0, word) for word in word_list]  # token index is ignored.
            tagged_token_dict[tag] = tagged_token_dict[tag].union(token_list)
    tagged_size_dict = {tag: len(token_set) for (tag, token_set) in tagged_token_dict.items()}
    return (tagged_token_dict, tagged_size_dict)


def get_tagged_size_dict(
    gold_list: List[str],
    pred_list: List[str],
    ignore_index: bool,
    is_extraction: bool = False,
    re_tag_paren: Tuple[str, str] = DEFAULT_TAG_PAREN_RE,
) -> Dict[str, Tuple[int, int, int]]:

    if not is_extraction:
        (gold_tagged_token_dict, gold_tagged_size_dict) = get_tagged_token_size_dict(gold_list)
        (pred_tagged_token_dict, pred_tagged_size_dict) = get_tagged_token_size_dict(pred_list)
    else:
        (gold_tagged_token_dict, gold_tagged_size_dict) = get_tagged_token_size_dict_extraction(gold_list, re_tag_paren)
        (pred_tagged_token_dict, pred_tagged_size_dict) = get_tagged_token_size_dict_extraction(pred_list, re_tag_paren)

    assert pred_tagged_token_dict.keys() == gold_tagged_token_dict.keys()

    intersection_tagged_token_dict = {
        tag: get_intersection(gold_tagged_token_dict[tag], pred_tagged_token_dict[tag], ignore_index)
        for tag in gold_tagged_token_dict.keys()
    }
    intersection_tagged_size_dict: Dict[str, int] = {
        tag: len(st) for (tag, st) in intersection_tagged_token_dict.items()
    }
    tag_key_set = gold_tagged_token_dict.keys()
    tag_size_dict: Dict[str, Tuple[int, int, int]] = {
        tag: (gold_tagged_size_dict[tag], pred_tagged_size_dict[tag], intersection_tagged_size_dict[tag])
        for tag in tag_key_set
    }
    # TODO: for each sentence (sample), TPR, FPR, etc. must be defined with equal weights.
    # TODO: later, those are averaged over the sentences (samples).
    # TODO: how about tags?
    #         average_options = {None, "micro", "macro", "weighted"}
    # TODO: TP, FP, TN, FN of this adjusted version can be regarded as
    # TODO: continuous extention of the discrete original TP, FP, TN, FN for one sample.
    # TODO: Therefore, one just need to sum up these to compute Precision and Recall.
    # TODO: https://atmarkit.itmedia.co.jp/ait/articles/2212/19/news020.html
    # TODO: macro-avg: average F1_type over all the tag types.
    # TODO: micro-avg: define TP, FP, TN, FN as sum of all the classes. micro-F1 == accuracy.
    return tag_size_dict


def tokenize_extraction(text: str, re_tag_paren: Tuple[str, str] = DEFAULT_TAG_PAREN_RE) -> List[str]:
    # TODO: Better to introduce a sophisticated tokenizer to support (multilingal) natural texts.
    delim = ","
    tag_paren1 = re_tag_paren[1].strip("\\")
    text_tmp1 = text.strip()
    text_tmp0 = re.sub(re_tag_paren[1] + delim, tag_paren1 + tag_paren1 + delim, text_tmp1)
    extracted_list = text_tmp0.split(tag_paren1 + delim)
    n_extracted_list = [e.strip() for e in extracted_list]
    return n_extracted_list


# def get_tagged_size_dict_extraction(
#     gold_list: List[str], pred_list: List[str], ignore_index: bool
# ) -> Dict[str, Tuple[int, int, int]]:

#     (gold_tagged_token_dict, gold_tagged_size_dict) = get_tagged_token_size_dict_extraction(gold_list)
#     (pred_tagged_token_dict, pred_tagged_size_dict) = get_tagged_token_size_dict_extraction(pred_list)

#     assert pred_tagged_token_dict.keys() == gold_tagged_token_dict.keys()

#     intersection_tagged_token_dict = {
#         tag: get_intersection(gold_tagged_token_dict[tag], pred_tagged_token_dict[tag], ignore_index)
#         for tag in gold_tagged_token_dict.keys()
#     }
#     intersection_tagged_size_dict: Dict[str, int] = {
#         tag: len(st) for (tag, st) in intersection_tagged_token_dict.items()
#     }
#     tag_key_set = gold_tagged_token_dict.keys()
#     tag_size_dict: Dict[str, Tuple[int, int, int]] = {
#         tag: (gold_tagged_size_dict[tag], pred_tagged_size_dict[tag], intersection_tagged_size_dict[tag])
#         for tag in tag_key_set
#     }
#     return tag_size_dict


def get_value_list(stats_dict: Dict[MetricName, Stat], prefix: str, tag: str, split: Union[str, None]) -> List[int]:
    value_name_list = ["gold", "pred", "intersection"]
    metric_name_list = [MetricName(prefix + "." + tag + "." + name, split=split) for name in value_name_list]
    value_list = [int(stats_dict[mn].sum) for mn in metric_name_list]
    return value_list


def compute_prrcf1(tp, tn, fp, fn) -> Tuple[float, float, float]:
    precision = float(tp) / float(tp + fp) if (tp + fp) != 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0.0 else 0.0
    return (precision, recall, f1)


def compute_tptnfpfn_adjusted(stats: List[int], total_token_length: int) -> Tuple[float, float, float, float]:
    assert len(stats) == 3
    ngold = stats[0]
    npred = stats[1]
    ninter = stats[2]
    # ltoken = total_token_length
    tp = float(ninter) / float(ngold) if ngold != 0.0 else 0.0
    fn = 1.0 - tp
    fp = float(npred - ninter) / float(npred) if npred != 0.0 else 0.0
    tn = 0.0  # not used.
    return (tp, tn, fp, fn)


def compute_tptnfpfn_modified_adjusted(stats: List[int], total_token_length: int) -> Tuple[float, float, float, float]:
    assert len(stats) == 3
    ngold = stats[0]
    npred = stats[1]
    ninter = stats[2]
    ltoken = total_token_length
    tp = ninter
    fn = ngold - ninter
    fp = npred - ninter
    tn = ltoken - (ngold + npred) + ninter
    return (tp, tn, fp, fn)


def compute_adjusted_f1(
    tag_stats_dict: Dict[str, List[int]], total_token_length: int, compute_tptnfpfn: Callable
) -> float:

    tag_tptnfpfn_list = [compute_tptnfpfn(stats, total_token_length) for stats in tag_stats_dict.values()]
    tag_prrcf1_list = [compute_prrcf1(v[0], v[1], v[2], v[3]) for v in tag_tptnfpfn_list]
    tag_f1_list = [v[2] for v in tag_prrcf1_list]
    macro_f1 = statistics.mean(tag_f1_list)

    return macro_f1


class NERAdjustedF1Metric(Metric):
    """
    Paper:
    Deußer, Tobias, et al.
    "KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from
    Financial Documents."
    arXiv preprint arXiv:2210.09163 (2022).
    https://arxiv.org/abs/2210.09163
    """

    NAME = "kpi_edgar_adjusted_f1"
    ignore_index = True
    is_extraction = True
    re_tag_paren = TAG_PAREN_RE

    def __init__(self):
        super().__init__()

        return

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Evaluate free-form generation."""

        # logging.warning(adapter_spec)
        # logging.warning("evaluate_generation instance.id: %s" % (request_state.instance.id))
        # logging.warning("evaluate_generation instance.reference: %s" % (request_state.instance.references))
        # logging.warning("evaluate_generation result.completion: %s" % (request_state.result.completions))
        # logging.warning(metric_service)

        golds: List[Reference] = [reference for reference in request_state.instance.references if reference.is_correct]
        completions: List[Sequence] = (
            cast(List[Sequence], request_state.result.completions) if request_state.result is not None else []
        )
        preds: List[str] = [completion.text.strip() for completion in completions]
        # logging.warning("evaluate_genearation len(preds): %d" % (len(preds)))
        # logging.warning("evaluate_genearation len(golds): %d" % (len(golds)))

        assert len(preds) >= 1
        assert len(golds) >= 1

        pred_text = preds[0]
        gold_text = golds[0].output.text.strip()
        # logging.warning("evaluate_genearation pred_text: %s" % (pred_text))
        # logging.warning("evaluate_genearation gold_text: %s" % (gold_text))

        pred_token_list = (
            tokenize(pred_text) if not self.is_extraction else tokenize_extraction(pred_text, self.re_tag_paren)
        )
        gold_token_list = (
            tokenize(gold_text) if not self.is_extraction else tokenize_extraction(gold_text, self.re_tag_paren)
        )
        gold_len = len(gold_token_list)
        # TODO: if the length are different, then the score should be 0.
        tagged_size_dict = get_tagged_size_dict(
            gold_token_list, pred_token_list, self.ignore_index, self.is_extraction, self.re_tag_paren
        )
        tag_stat_tpl_list = [
            (
                Stat(MetricName(self.NAME + "." + tag + "." + "gold")).add(vals[0]),
                Stat(MetricName(self.NAME + "." + tag + "." + "pred")).add(vals[1]),
                Stat(MetricName(self.NAME + "." + tag + "." + "intersection")).add(vals[2]),
            )
            for (tag, vals) in tagged_size_dict.items()
        ]
        tag_stat_list = list(itertools.chain.from_iterable(tag_stat_tpl_list))
        len_stat = Stat(MetricName(self.NAME + "." + "len")).add(gold_len)
        return tag_stat_list + [len_stat]

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        """Evaluate all request states directly. Use only if nothing else works.  Override me!"""

        # logging.warning("evaluate_instances len: %d" % (len(request_states)))
        # logging.warning(request_states[0].instance.id)

        return []

    def derive_stats(self, stats_dict: Dict[MetricName, Stat]) -> List[Stat]:
        """Derive stats based on existing stats, e.g., for perplexity. Override me!"""

        # logging.warning("derive_stats stats_dict: %s" % (stats_dict))
        # TODO:
        # I assume that all the stats in stats_dict were computed from the same split (valid or test).
        assert len(stats_dict) >= 1
        stats_list: List[Stat] = list(stats_dict.values())

        stat = stats_list[0]
        split = stat.name.split
        # logging.warning("derive_stats stat: %s" % (stat))
        logging.warning("derive_stats split: %s" % (split))

        tag_stat_dict: Dict[str, List[int]] = {
            tag: get_value_list(stats_dict, self.NAME, tag, split) for tag in TAG_DICT.keys()
        }
        # logging.warning(tag_stat_dict)
        stats_total_token = stats_dict[MetricName(self.NAME + "." + "len", split=split)]
        adjusted_f1 = compute_adjusted_f1(tag_stat_dict, int(stats_total_token.sum), compute_tptnfpfn_adjusted)
        # logging.warning("derive_stats adjusted_f1: %f" % (adjusted_f1))
        modified_adjusted_f1 = compute_adjusted_f1(
            tag_stat_dict, int(stats_total_token.sum), compute_tptnfpfn_modified_adjusted
        )
        # logging.warning("derive_stats modified adjusted_f1: %f" % (modified_adjusted_f1))
        return [
            Stat(MetricName(self.NAME + "." + "macro", split=split)).add(adjusted_f1),
            Stat(MetricName(self.NAME + "." + "modified_macro", split=split)).add(modified_adjusted_f1),
        ]
