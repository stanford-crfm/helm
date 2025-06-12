from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG


class RulerQAMetric(Metric):
    """Accuracy metric for Ruler QA Scenarios.

    Adapted from: https://github.com/NVIDIA/RULER/blob/1c45e5c60273e0ae9e3099137bf0eec6f0395f84/scripts/eval/synthetic/constants.py#L25
    """  # noqa: E501

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result
        assert len(request_state.result.completions) == 1
        response_text = request_state.result.completions[0].text
        assert all(
            len(reference.tags) == 1 and reference.tags[0] == CORRECT_TAG
            for reference in request_state.instance.references
        )
        reference_texts = [reference.output.text for reference in request_state.instance.references]
        score = max([1.0 if r.lower() in response_text.lower() else 0.0 for r in reference_texts])
        return [Stat(MetricName("ruler_string_match_part")).add(score)]
