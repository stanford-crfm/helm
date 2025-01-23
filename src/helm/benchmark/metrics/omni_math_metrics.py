from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class OmniMATHMetric(Metric):
    """Score metrics for Omni-MATH."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.annotations
        all_judgements = request_state.annotations["omni_math"]["equivalence_judgement"]
        if len(all_judgements) == 0:
            raise ValueError("Could not compute Omni-MATH accuracy because all annotators failed.")
        judgement_bools = [judgement.strip().upper() == "TRUE" for judgement in all_judgements]
        score = sum(judgement_bools) / len(judgement_bools)
        return [
            Stat(MetricName("omni_math_accuracy")).add(score),
        ]
