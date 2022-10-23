import pandas
import numpy as np
import spacy
import subprocess
import sys
import os
from typing import List, Dict
from collections import defaultdict
from common.general import ensure_file_downloaded

# Need to check spacy module is downloaded before importing DataStatsMetric
if not spacy.util.is_package("en_core_web_sm"):
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

from summ_eval.data_stats_metric import DataStatsMetric

from benchmark.adapter import AdapterSpec, RequestState, ScenarioState
from common.hierarchical_logger import hlog
from .metric import Metric, MetricResult
from .metric_name import MetricName
from .metric_service import MetricService
from .basic_metrics import get_rouge_function
from .statistic import Stat
from .summac.model_summac import SummaCZS
from bert_score import BERTScorer

class SummarizationMetric(Metric):
    """Summarization Metrics

    This class computes the following standard summarization metrics
        1. Rouge (1,2,L)
        2. Extractiveness (coverage, density, novel n-grams)
        3. Compression
        4. Faithfulness (SummaC)
    """

    def __init__(self, device="cpu"):
        self.rouge_fns = {
            "rouge_1": get_rouge_function("rouge1"),
            "rouge_2": get_rouge_function("rouge2"),
            "rouge_l": get_rouge_function("rougeL"),
        }
        self.data_stats_metric = DataStatsMetric()

        if device == "cpu":
            self.compute_faithfulness = False
            self.compute_bertscore = False
        else:
            self.compute_bertscore = True
            self.bert_scorer = BERTScorer(
                model_type="microsoft/deberta-large-mnli", lang="en", rescale_with_baseline=True, device=device
            )
            # Need GPU for faithfulness metrics since they are model-based.
            self.compute_faithfulness = True
            self.summac = SummaCZS(granularity="sentence", model_name="vitc", imager_load_cache=False, device=device)

    def evaluate(
        self, scenario_state: ScenarioState, metric_service: MetricService, eval_cache_path: str, parallelism: int
    ) -> MetricResult:
        if self.compute_faithfulness:
            # When running with a GPU and parallelism > 1, errors with "...in layer_norm
            # return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
            # RuntimeError: expected scalar type Float but found Half".
            hlog(
                f"Setting parallelism from {parallelism} to 1, since "
                f"evaluating faithfulness with parallelism > 1 errors."
            )
            parallelism = 1

        return super().evaluate(scenario_state, metric_service, eval_cache_path, parallelism=parallelism)

    def _compute_rouge(self, refs: List[str], pred: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for metric, metric_fn in self.rouge_fns.items():
            metrics[metric] = np.max([metric_fn(ref, pred) for ref in refs])

        return metrics

    def _compute_data_stats(self, inp: str, pred: str) -> Dict[str, float]:
        stats = self.data_stats_metric.evaluate_example(pred, inp)
        return {
            "summarization_coverage": stats["coverage"],
            "summarization_density": stats["density"],
            "summarization_compression": stats["compression"],
        }

    def _compute_faithfulness_scores(self, inp: str, pred: str) -> Dict[str, float]:
        return {"summac": self.summac.score_one(inp, pred)["score"]}

    def _compute_bert_score(self, refs: List[str], pred: str) -> Dict[str, float]:
        p, r, f = self.bert_scorer.score([pred], [refs])
        return {"BERTScore-P": p[0].item(), "BERTScore-R": r[0].item(), "BERTScore-F": f[0].item()}

    def _remove_braces(self, text):
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

        refs: List[str] = [self._remove_braces(x.output) for x in request_state.instance.references]
        inp: str = self._remove_braces(request_state.instance.input)

        assert request_state.result is not None
        pred: str = self._remove_braces(request_state.result.completions[0].text.strip())

        result: List[Stat] = []

        # Compute rouge metrics
        result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_rouge(refs, pred).items()])

        # Compute data stats
        result.extend(
            [Stat(MetricName(name)).add(float(val)) for name, val in self._compute_data_stats(inp, pred).items()]
        )

        # Compute faithfulness metric(s)
        if self.compute_faithfulness:
            result.extend(
                [
                    Stat(MetricName(name)).add(float(val))
                    for name, val in self._compute_faithfulness_scores(inp, pred).items()
                ]
            )

        # Compute BERTScore
        if self.compute_bertscore:
            result.extend(
                [Stat(MetricName(name)).add(float(val)) for name, val in self._compute_bert_score(refs, pred).items()]
            )

        return result

HUMAN_EVAL_CODALAB_LINK: str = (
    "https://worksheets.codalab.org/rest/bundles/0x8c5eeb13c0bd47b1b4a74791f4a38425/contents/blob/summ_humaneval/{file_name}"
)

def _paired_bootstrap_test(treatment, control, nboot=10000):
    treatment = np.array(treatment)
    control = np.array(control)
    delta = treatment.mean() - control.mean()
    sample_idx = np.random.choice(np.arange(len(treatment)), size=(nboot, len(treatment)))
    boot_treatment = treatment[sample_idx]
    boot_control = control[sample_idx]
    diff = boot_treatment.mean(axis=1) - boot_control.mean(axis=1)
    return (diff > 2 * delta).mean()


class SummarizationHumanEvalAnalyzer():
    """
    Analyzes the human evaluation data of on summarization datasets

    1. loads humaneval data from codaalb
    2. averages and report {faithfulness, relevance, coherence} scores
    3. compute paired bootstrap test for all pairwise model comparison
    """

    def __init__(self, dataset:str, eval_download_path:str):
        self.dataset = dataset
        self.eval_download_path = eval_download_path
        os.makedirs(eval_download_path, exist_ok=True)
        self.faithfulness, self.coherence, self.relevance = None, None, None

    def load_humaneval_data(self):
        filenames = [
            f'Batch_{self.dataset}_small_results.csv',
            f'Batch_{self.dataset}_large_results.csv'
        ]

        tasks_by_id = defaultdict(list)

        for filename in filenames:
            download_filename = HUMAN_EVAL_CODALAB_LINK.format(file_name=filename)
            filename = os.path.join(self.eval_download_path, filename)
            ensure_file_downloaded(source_url=download_filename, target_path=filename)
            mturk_data = pandas.read_csv(filename)
            for i, row in mturk_data.iterrows():
                tasks_by_id[row.HITId].append(row)

        self.faithfulness = defaultdict(list)
        for idx, tasks in tasks_by_id.items():
            scores = []
            for task in tasks:
                scores.append(1 if task['Answer.consistency.consistent'] else 0)
            self.faithfulness[task['Input.model_name']].append(np.mean(scores))

        self.coherence = defaultdict(list)
        for idx, tasks in tasks_by_id.items():
            scores = []
            for task in tasks:
                for i in range(1, 6):
                    if task[f'Answer.coherence.cohere_{i}']:
                        scores.append(i)
                        break
            self.coherence[task['Input.model_name']].append(np.mean(scores))

        self.relevance = defaultdict(list)
        for idx, tasks in tasks_by_id.items():
            scores = []
            for task in tasks:
                for i in range(1, 6):
                    if task[f'Answer.relevance.rel_{i}']:
                        scores.append(i)
                        break
            self.relevance[task['Input.model_name']].append(np.mean(scores))

    def _compute_average(self, scores: dict):
        return [(x, np.mean(y)) for x, y in scores.items()]

    def print_summary(self):
        assert self.faithfulness
        assert self.coherence
        assert self.relevance

        print("FAITHFULNESS")
        for model, score in self._compute_average(self.faithfulness):
            print(f"{model:40}: {score:.4f}")
        print("="*40)

        print("RELEVANCE")
        for model, score in self._compute_average(self.relevance):
            print(f"{model:40}: {score:.4f}")
        print("="*40)

        print("COHERENCE")
        for model, score in self._compute_average(self.relevance):
            print(f"{model:40}: {score:.4f}")
        print("="*40)

    def print_test_result(self):
        assert self.faithfulness
        assert self.coherence
        assert self.relevance

        print("FAITHFULNESS")
        avg_faithful_scores = self._compute_average(self.faithfulness)
        sorted_models, _ = zip(*sorted(avg_faithful_scores, key=lambda x:x[1], reverse=True))
        for i, best_model in enumerate(sorted_models):
            for other_model in sorted_models[i+1:]:
                p_value = _paired_bootstrap_test(self.faithfulness[best_model], self.faithfulness[other_model])
                print(f"{best_model} > {other_model}: p-value {p_value:.3f}")
        print("="*40)

        print("RELEVANCE")
        avg_faithful_scores = self._compute_average(self.relevance)
        sorted_models, _ = zip(*sorted(avg_faithful_scores, key=lambda x: x[1], reverse=True))
        for i, best_model in enumerate(sorted_models):
            for other_model in sorted_models[i + 1:]:
                p_value = _paired_bootstrap_test(self.relevance[best_model], self.relevance[other_model])
                print(f"{best_model} > {other_model}: p-value {p_value:.3f}")
        print("=" * 40)

        print("COHERENCE")
        avg_faithful_scores = self._compute_average(self.coherence)
        sorted_models, _ = zip(*sorted(avg_faithful_scores, key=lambda x: x[1], reverse=True))
        for i, best_model in enumerate(sorted_models):
            for other_model in sorted_models[i + 1:]:
                p_value = _paired_bootstrap_test(self.coherence[best_model], self.coherence[other_model])
                print(f"{best_model} > {other_model}: p-value {p_value:.3f}")
        print("=" * 40)