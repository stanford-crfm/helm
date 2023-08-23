from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from .metric import Metric
from .metric_name import MetricName
from .statistic import Stat
from nltk.translate.bleu_score import corpus_bleu


class CLEVAParaphraseGenerationMetric(Metric):
    """
    Compute the Chinese iBLEU score for Paraphrase Generation scenarios of CLEVA benchmark.

    Reference:
    https://aclanthology.org/2022.acl-long.178.pdf
    https://aclanthology.org/P12-2008.pdf
    """

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:

        inputs: List = []
        golds: List = []
        preds: List = []

        for request_state in request_states:
            inputs.append(request_state.instance.input.text)

            golds.append(request_state.instance.references[0].output.text)

            assert request_state.result is not None
            preds.append(request_state.result.completions[0].text)

        alpha = 0.8  # to calculate ibleu0.8
        bleu = corpus_bleu([[i for i in golds]], [[i for i in preds]], weights=(1, 0, 0, 0))
        sbleu = corpus_bleu([[i for i in inputs]], [[i for i in preds]], weights=(1, 0, 0, 0))
        chinese_ibleu_score = alpha * bleu - (1 - alpha) * sbleu

        return [Stat(MetricName("chinese_ibleu")).add(chinese_ibleu_score)]
