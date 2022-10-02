import numpy as np
import spacy
import subprocess
import sys
from typing import List, Dict

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

    def _compute_bert_score(self, inp: str, pred: str) -> Dict[str, float]:
        p, r, f = self.bert_scorer.score([inp], [pred])
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
                [Stat(MetricName(name)).add(float(val)) for name, val in self._compute_bert_score(inp, pred).items()]
            )

        return result
