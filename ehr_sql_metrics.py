from typing import List, Dict, Any
import re
import sqlite3
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


class EhrSqlMetric(Metric):
    """
    Metric for evaluating the EHR SQL dataset, focusing on:
    1. Execution Accuracy – Whether the generated SQL query produces the same results as the ground truth.
    2. Query Validity – Whether the generated SQL executes without errors.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate execution accuracy of generated SQL queries.
        """

        if not request_state.annotations:
            raise Exception("Request state does not contain annotations.")

        # Extract execution results
        predicted_result = request_state.annotations["ehr_sql"]["predicted_result"]
        ground_truth_result = request_state.annotations["ehr_sql"]["ground_truth_result"]
        query_error = request_state.annotations["ehr_sql"]["query_error"]

        # Determine execution accuracy
        execution_accuracy = int(set(predicted_result) == set(ground_truth_result))
        query_validity = 1 if query_error is None else 0

        return [
            Stat(MetricName("ehr_sql_execution_accuracy")).add(execution_accuracy),
            Stat(MetricName("ehr_sql_query_validity")).add(query_validity),
        ]

    def compute(self, stats: List[Stat], **kwargs: Any) -> Dict[str, float]:
        """
        Aggregate statistics to compute final metrics.
        """

        execution_correct = sum(
            stat.value for stat in stats if stat.name == "ehr_sql_execution_accuracy"
        )
        query_valid = sum(
            stat.value for stat in stats if stat.name == "ehr_sql_query_validity"
        )
        total_queries = len(stats) // 2  # Each query contributes two stats

        return {
            "ehr_sql_execution_accuracy": execution_correct / total_queries if total_queries > 0 else 0.0,
            "ehr_sql_query_validity": query_valid / total_queries if total_queries > 0 else 0.0,
        }
