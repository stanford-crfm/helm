from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class BirdSQLMetric(Metric):
    """Score metrics for Bird-SQL."""

    ANNOTATOR_NAME = "bird_sql"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        if not request_state.annotations:
            raise Exception("Request state did not have annotations.")
        predicted_result = request_state.annotations[self.ANNOTATOR_NAME]["predicted_result"]
        ground_truth_result = request_state.annotations[self.ANNOTATOR_NAME]["ground_truth_result"]
        execution_accuracy = int(set(predicted_result) == set(ground_truth_result))
        return [Stat(MetricName("execution_accuracy")).add(execution_accuracy)]
