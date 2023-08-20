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

        refs: List[List[str]] = [[]]
        sys: List = []
        for request_state in request_states:
            # Assume there is one referece per instance. TODO: Support multiple references after adding more scenarios.
            num_references: int = len(request_state.instance.references)
            if num_references != 1:
                raise ValueError(f"This instance has {num_references} references, but we currently only support one.")
            # Usually there is only one completion for each instance.
            assert request_state.result is not None
            if len(request_state.result.completions) != 1:
                raise ValueError("Each request result should have only exactly one completion.")
            sys.append(request_state.result.completions[0].text)
            refs[0].append(request_state.instance.references[0].output.text)
        bleu_score = bleu.corpus_score(sys, refs).score
        return [Stat(MetricName("bleu")).add(bleu_score)]


class ChineseMachineTranslationMetric(Metric):
    """
    Compute the BLEU score for Machine Translation scenarios whose target language is Chinese.
    The implementation is based on sacrebleu.
    """

    def evaluate_instances(self, request_states: List[RequestState]) -> List[Stat]:
        """
        Compute the corpus-level metric based on all reqeust_states.
        """

        bleu = BLEU(tokenize="zh")

        refs: List[List[str]] = [[]]
        sys: List = []
        max_num_references = max([len(request_state.instance.references) for request_state in request_states])
        refs = [[l[i] if i < len(l) else "" for l in label] for i in range(max_num_references)]
        for request_state in request_states:
            # Assume there is one referece per instance. TODO: Support multiple references after adding more scenarios.
            num_references: int = len(request_state.instance.references)
            if num_references != 1:
                raise ValueError(f"This instance has {num_references} references, but we currently only support one.")
            # Usually there is only one completion for each instance.
            assert request_state.result is not None
            if len(request_state.result.completions) != 1:
                raise ValueError("Each request result should have only exactly one completion.")
            sys.append(request_state.result.completions[0].text)
            refs[0].append(request_state.instance.references[0].output.text)
        bleu_score = bleu.corpus_score(sys, refs).score
        return [Stat(MetricName("bleu")).add(bleu_score)]
