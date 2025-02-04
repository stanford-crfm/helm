import numbers
from typing import Any, Dict, List, cast

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.basic_metrics import compute_request_state_metrics
from helm.benchmark.metrics.efficiency_metrics import EfficiencyMetric
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

class BirdSQLMetric(Metric):
    """Score metrics for HELM Safety."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # For now, assume there is only one annotator.

        if not request_state.annotations:
            raise Exception("Request state did not have annotations.")
        predicted_result = request_state.annotations["bird_sql"]["predicted_result"]
        ground_truth_result = request_state.annotations["bird_sql"]["ground_truth_result"]
        execution_accuracy = int(set(predicted_result) == set(ground_truth_result))
        return [
                Stat(MetricName(f"execution_accuracy")).add(execution_accuracy)
        ]
