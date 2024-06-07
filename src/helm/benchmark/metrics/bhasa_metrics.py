import re
import string
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Dict, List, cast
from collections import Counter

import numpy as np
from nltk.metrics.scores import f_measure
from pythainlp.tokenize import word_tokenize
from sacrebleu.metrics import CHRF

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.evaluate_reference_metrics import exact_match
from helm.benchmark.metrics.evaluate_reference_metrics import rouge_score as rouge_score_fn
from helm.benchmark.metrics.metric import Metric, MetricResult, MetricSpec
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

class BhasaMachineTranslationMetric(Metric):
    """Machine Translation Metrics

    This class computes the following standard machine translation metrics

    1. ChrF++

    @inproceedings{popovic-2015-chrf,
        title = "chr{F}: character n-gram {F}-score for automatic {MT} evaluation",
        author = "Popovi{\'c}, Maja",
        editor = "Bojar, Ond{\v{r}}ej  and
        Chatterjee, Rajan  and
        Federmann, Christian  and
        Haddow, Barry  and
        Hokamp, Chris  and
        Huck, Matthias  and
        Logacheva, Varvara  and
        Pecina, Pavel",
        booktitle = "Proceedings of the Tenth Workshop on Statistical Machine Translation",
        month = sep,
        year = "2015",
        address = "Lisbon, Portugal",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/W15-3049",
        doi = "10.18653/v1/W15-3049",
        pages = "392--395",
        github = "https://github.com/mjpost/sacrebleu",
    }
    """

    def __init__(self):
        self.chrf_scorer = CHRF(word_order=2)

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def _compute_chrf(self, refs: List[str], pred: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics['ChrF++'] = self.chrf_scorer.sentence_score(pred, refs).score
        return metrics

    def _remove_braces(self, text: str) -> str:
        if text.startswith("{"):
            text = text[1:]
        if text.endswith("}"):
            text = text[:-1]
        return text

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        refs: List[str] = [self._remove_braces(ref.output.text) for ref in request_state.instance.references]
        inp: str = self._remove_braces(request_state.instance.input.text)

        assert request_state.result is not None
        pred: str = self._remove_braces(request_state.result.completions[0].text.strip())

        result: List[Stat] = []

        # Compute ChrF++ metrics
        result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_chrf(refs, pred).items()])

        return result

class BhasaQAMetric(Metric):
    """Bhasa QA Metrics

    This class computes the following standard SQuAD v1.1 metrics

    1. SQuAD exact match
    2. SQuAD macro-averaged F1 score

    @inproceedings{rajpurkar-etal-2016-squad,
        title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
        author = "Rajpurkar, Pranav  and
            Zhang, Jian  and
            Lopyrev, Konstantin  and
            Liang, Percy",
        editor = "Su, Jian  and
            Duh, Kevin  and
            Carreras, Xavier",
        booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
        month = nov,
        year = "2016",
        address = "Austin, Texas",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/D16-1264",
        doi = "10.18653/v1/D16-1264",
        pages = "2383--2392",
    }
    """

    def __init__(self, language: str = 'en'):
        self.language: str = language
        self.metrics: Dict[str, Callable] = {
            "squad_exact_match_score": self.squad_exact_match_score,
            "squad_f1_score": self.squad_f1_score,
        }

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def normalize_answer(self, text: str) -> List[str]:
        """
        For Thai, this will split the text using PyThaiNLP's tokenizer.
        For all other languages, this will:
        - Lower text
        - Remove punctuation
        - Remove extra whitespace

        If the language is English, it will
        - Remove articles "a", "an", and "the"

        Modifies code from [SQuAD v1.1](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py).
        """

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
        if self.language == "th":
            return word_tokenize(normalized_text, engine="newmm")
        elif self.language == "en":
            return white_space_fix(remove_articles(normalized_text)).split()
        else:
            return white_space_fix(normalized_text).split()

    def squad_f1_score(self, gold: str, pred: str) -> float:
        prediction_tokens = self.normalize_answer(pred)
        ground_truth_tokens = self.normalize_answer(gold)
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def squad_exact_match_score(self, gold: str, pred: str) -> float:
        return self.normalize_answer(pred) == self.normalize_answer(gold)

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:

        stats: List[Stat] = []
        if len(request_state.instance.references) > 0:
            golds = [reference for reference in request_state.instance.references if reference.is_correct]
            assert len(golds) > 0

            assert request_state.result is not None
            sorted_completions = sorted(request_state.result.completions, key=lambda x: -x.logprob)
            preds = [completion.text.strip() for completion in sorted_completions]

            for name, metric in self.metrics.items():
                name = MetricName(name)
                metric = cast(Callable[[str, str], float], metric)
                score_1 = max(metric(gold.output.text.strip(), preds[0]) for gold in golds)
                score_k = max(metric(gold.output.text.strip(), pred) for gold in golds for pred in preds)

                metrics = [Stat(name).add(score_1)]
                if adapter_spec.num_outputs != 1:
                    metrics.append(Stat(replace(name, name=f"{name.name}@{adapter_spec.num_outputs}")).add(score_k))
                stats.extend(metrics)

        return stats

def get_bhasa_machine_translation_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.bhasa_metrics.BhasaMachineTranslationMetric")
    ]

def get_bhasa_qa_metric_specs(args: Dict[str, Any]) -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.bhasa_metrics.BhasaQAMetric", args=args)
    ]
