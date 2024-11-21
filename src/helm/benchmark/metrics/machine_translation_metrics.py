from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat

try:
    from sacrebleu.metrics import BLEU
    from langdetect import detect
except ModuleNotFoundError as e:
    handle_module_not_found_error(e)


class MachineTranslationMetric(EvaluateInstancesMetric):
    """
    Compute the BLEU score for Machine Translation scenarios. The implementation is based on sacrebleu.
    """

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
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


class CLEVAMachineTranslationMetric(EvaluateInstancesMetric):
    """
    Compute the BLEU score for Machine Translation scenarios of CLEVA benchmark.
    Based on sacrebleu, this implementation distinguishes target language and allows variable number of references.
    If there are more than one hypothesis, only the first one is adopted in the calculation.
    """

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        """
        Compute the corpus-level metric based on all reqeust_states.
        """

        def detect_language(request_states: List[RequestState]) -> str:
            """
            Determine the target language by detecting the language of references.
            Currently, it only distinguishes if the target language is Chinese.
            """

            corpus: str = "".join(
                [request_state.instance.references[0].output.text for request_state in request_states[:10]]
            )
            if detect(corpus) in ["zh-cn", "zh-tw"]:
                return "zh"
            else:
                return "13a"  # Default tokenizer for sacrebleu.BLEU

        bleu = BLEU(tokenize=detect_language(request_states))

        max_num_references: int = max([len(request_state.instance.references) for request_state in request_states])
        refs: List[List[str]] = [
            [
                request_state.instance.references[i].output.text if i < len(request_state.instance.references) else ""
                for request_state in request_states
            ]
            for i in range(max_num_references)
        ]

        sys: List = []
        for request_state in request_states:
            assert request_state.result is not None
            sys.append(request_state.result.completions[0].text)

        bleu_score = bleu.corpus_score(sys, refs).score

        return [Stat(MetricName("cleva_machine_translation_bleu")).add(bleu_score)]
