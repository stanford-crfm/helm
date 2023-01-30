from typing import List
from sacrebleu import BLEU

from helm.benchmark.adaptation.request_state import RequestState
from .metric import Metric
from .metric_name import MetricName
from .statistic import Stat


class MachineTranslationMetric(Metric):
    """
    Compute the BLEU score for Machine Translation scenarios. The implementation is based on sacrebleu.
    """

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        """
        Compute the corpus-level metric based on all reqeust_states.
        """

        bleu = BLEU()

        # Assume there is one referece for each instance. TODO: Support multiple references after adding more scenarios.
        refs: List[List[str]] = [[]]
        sys: List = []
        for request_state in request_states:
            # Usually there is only one completion for each instance.
            assert request_state.result is not None
            sys.append(request_state.result.completions[0].text)
            refs[0].append(request_state.instance.references[0].output.text)
        bleu_score = bleu.corpus_score(sys, refs).score
        return [Stat(MetricName("bleu")).add(bleu_score)]
