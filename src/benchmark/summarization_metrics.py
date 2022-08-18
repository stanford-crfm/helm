import numpy as np
import spacy
import subprocess
import sys

# Need to check spacy module is downlaoded before importing DataStatsMetric
if not spacy.util.is_package("en_core_web_sm"):
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

from .statistic import Stat
from typing import List, Sequence, Dict
from summ_eval.data_stats_metric import DataStatsMetric

from .adapter import AdapterSpec, RequestState
from .scenarios.scenario import Reference
from .metric import Metric
from .metric_name import MetricName
from .metric_service import MetricService
from .basic_metrics import get_rouge_function
from .metrics.summac.model_summac import SummaCZS


class SummarizationMetric(Metric):
    """Summarizaitron Metrics

    This class computes the following standard summarization metrcis
        1. Rouge (1,2,L)
        2. Extractiveness (coverage, density, novel n-grams)
        3. Compression
        4. Faithfulness (SummaC)
    """

    def __init__(self, device="cpu"):
        self.rouge_fns = {
            "rouge-1": get_rouge_function("rouge1"),
            "rouge-2": get_rouge_function("rouge2"),
            "rouge-l": get_rouge_function("rougeL"),
        }
        self.data_stats_metric = DataStatsMetric()
        self.summac = SummaCZS(granularity="sentence", model_name="vitc", imager_load_cache=False)

    def _compute_rouge(self, refs: Sequence[Reference], pred: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for metric, metric_fn in self.rouge_fns.items():
            metrics[metric] = np.max([metric_fn(ref.output, pred) for ref in refs])

        return metrics

    def _compute_data_stats(self, inp: str, pred: str) -> Dict[str, float]:
        stats = self.data_stats_metric.evaluate_example(pred, inp)
        return {"coverage": stats["coverage"], "density": stats["density"], "compression": stats["compression"]}

    def _compute_faithfulness_scores(self, inp: str, pred: str) -> Dict[str, float]:
        return {"SummaC": self.summac.score_one(inp, pred)["score"]}

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """Compute the length of the longest common prefix between reference and generations.

        Result is based on number of tokens produced with `nltk.tokenize.TreebankWordTokenizer`.
        When there are multiple generations, return the length of the longest.

        Example:
            input: A
            generations: [A A B C, A M D]
            reference: A A D
            returns: 2
            explanation: The longest common prefix is A A (between A A B C and A A D).
        """
        refs: Sequence[Reference] = request_state.instance.references
        inp: str = request_state.instance.input

        assert request_state.result is not None
        pred: str = request_state.result.completions[0].text.strip()

        result: List[Stat] = []

        # Compute rouge metrics
        result.extend([Stat(MetricName(name)).add(float(val)) for name, val in self._compute_rouge(refs, pred).items()])

        # Compute data stats
        result.extend(
            [Stat(MetricName(name)).add(float(val)) for name, val in self._compute_data_stats(inp, pred).items()]
        )

        # Compute faithfulness metric(s)
        result.extend(
            [
                Stat(MetricName(name)).add(float(val))
                for name, val in self._compute_faithfulness_scores(inp, pred).items()
            ]
        )

        return result
