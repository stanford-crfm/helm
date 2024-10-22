from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from nltk.translate.bleu_score import corpus_bleu


class CLEVAParaphraseGenerationMetric(EvaluateInstancesMetric):
    """
    Compute the Chinese iBLEU score for Paraphrase Generation scenarios of CLEVA benchmark.
    This implementation allows variable number of references (i.e., golds).
    If there are more than one hypothesis (i.e., preds), only the first one is adopted in the calculation.

    Reference:
    https://aclanthology.org/2022.acl-long.178.pdf
    https://aclanthology.org/P12-2008.pdf
    """

    def __init__(self, alpha: float = 0.8):  # calculate iBLEU_0.8 by default
        self.alpha = alpha

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        inputs: List = []
        preds: List = []
        golds: List[List[str]] = []

        for request_state in request_states:
            inputs.append(request_state.instance.input.text)

            assert request_state.result is not None
            preds.append(request_state.result.completions[0].text)

            golds.append([reference.output.text for reference in request_state.instance.references])

        # using characters for computing BLEU
        tokenized_inputs = [[[i for i in input]] for input in inputs]
        tokenized_preds = [[i for i in pred] for pred in preds]
        tokenized_golds = [[[i for i in gold] for gold in references] for references in golds]

        bleu = corpus_bleu(tokenized_golds, tokenized_preds, weights=(1, 0, 0, 0))
        sbleu = corpus_bleu(tokenized_inputs, tokenized_preds, weights=(1, 0, 0, 0))
        chinese_ibleu_score = self.alpha * bleu - (1 - self.alpha) * sbleu

        return [Stat(MetricName("chinese_ibleu")).add(chinese_ibleu_score)]
