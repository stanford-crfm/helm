import re
import string
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Dict, List, cast

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
from helm.benchmark.metrics.xlsum import rouge_scorer, tokenizers


class BhasaMachineTranslationMetric(Metric):
    """Machine Translation Metrics

    This class computes the following standard machine translation metrics

    1. ChrF++
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
    """

    def __init__(self, language: str = 'en'):
        self.language: str = language
        self.metrics: Dict[str, Callable] = {
            "squad_exact_match": exact_match,
            "squad_f1_score": self.squad_f1_score,
        }

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def split_text(self, text: str) -> List[str]:
        """
        For Thai, this will split the text using PyThaiNLP's tokenizer.
        For all other languages, this will:
        - Lower text
        - Remove punctuation
        - Remove extra whitespace

        If the language is English, it will
        - Remove articles "a", "an", and "the"

        Modifies code from [QuAC](http://quac.ai/) found at https://s3.amazonaws.com/my89public/quac/scorer.py
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
            return set(word_tokenize(normalized_text, engine="newmm"))
        elif self.language == "en":
            return set(white_space_fix(remove_articles(normalized_text)).split())
        else:
            return set(white_space_fix(normalized_text).split())

    def squad_f1_score(self, gold: str, pred: str) -> float:
        score = f_measure(self.split_text(gold), self.split_text(pred))

        if score is None:
            return 0.0
        return score

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

        stats: List[Stat] = []
        if len(request_state.instance.references) > 0:
            golds = [reference for reference in request_state.instance.references if reference.is_correct]
            assert len(golds) > 0

            assert request_state.result is not None
            sorted_completions = sorted(request_state.result.completions, key=lambda x: -x.logprob)
            preds = [self._remove_braces(completion.text).strip() for completion in sorted_completions]

            for name, metric in self.metrics.items():
                name = MetricName(name)
                metric = cast(Callable[[str, str], float], metric)
                score_1 = max(metric(self._remove_braces(gold.output.text).strip(), preds[0]) for gold in golds)
                score_k = max(metric(self._remove_braces(gold.output.text).strip(), pred) for gold in golds for pred in preds)

                metrics = [Stat(name).add(score_1)]
                if adapter_spec.num_outputs != 1:
                    metrics.append(Stat(replace(name, name=f"{name.name}@{adapter_spec.num_outputs}")).add(score_k))
                stats.extend(metrics)

        return stats

class BhasaSummarizationMetric(Metric):
    """Summarization Metrics

    This class computes the following standard summarization metrics

    1. Rouge-L (F1 score, using the "mid" result when performing bootstrap aggregation)
    """

    def __init__(self, language: str = 'en'):
        self.language: str = language
        self.rouge_fns = {
            "rouge_l": self._get_bhasa_rouge_function("rougeL"),
        }

    def _get_bhasa_rouge_function(self, rouge_type: str) -> Callable[[str, str], float]:
        if self.language == "th":
            scorer = rouge_scorer.RougeScorer(
                [rouge_type],
                use_stemmer=True,
                callable_tokenizer=tokenizers.ThaiTokenizer())
        else:
            scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        return partial(rouge_score_fn, scorer=scorer, rouge_type=rouge_type)

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def _compute_rouge(self, refs: List[str], pred: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for metric, metric_fn in self.rouge_fns.items():
            metrics[metric] = np.max([metric_fn(ref, pred) for ref in refs])

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

        # Compute rouge metrics
        result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_rouge(refs, pred).items()])

        return result

def get_bhasa_machine_translation_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.bhasa_metrics.BhasaMachineTranslationMetric")
    ]

def get_bhasa_summarization_metric_specs(args: Dict[str, Any]) -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.bhasa_metrics.BhasaSummarizationMetric", args=args)
    ]

def get_bhasa_qa_metric_specs(args: Dict[str, Any]) -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.bhasa_metrics.BhasaQAMetric", args=args)
    ]
