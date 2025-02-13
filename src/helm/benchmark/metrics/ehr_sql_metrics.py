from typing import List, Dict, Any
import re
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
    3. Precision for Answerable Questions (Pans).
    4. Recall for Answerable Questions (Rans).
    """

    def extract_is_impossible(self, input_text: str) -> bool:
        """Extracts `is_impossible` from input_text using regex."""
        match = re.search(r'"is_impossible":\s*(true|false)', input_text, re.IGNORECASE)
        return match and match.group(1).lower() == "true"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate execution accuracy, query validity, and answerability metrics.
        """

        if not request_state.annotations:
            raise Exception("Request state does not contain annotations.")

        # Extract execution results
        predicted_result = request_state.annotations["ehr_sql"]["predicted_result"]
        ground_truth_result = request_state.annotations["ehr_sql"]["ground_truth_result"]
        query_error = request_state.annotations["ehr_sql"]["query_error"]

        # Extract predictions from the model output
        predictions = [completion.text.strip() for completion in request_state.result.completions]

        if not predictions:
            raise ValueError("No predictions found in the completions.")

        # Process the first prediction as the primary output
        prediction = predictions[0].strip()

        # Extract references and input text
        references = getattr(request_state.instance, "references", None)
        input_text = request_state.instance.input.text  # Read input text

        if not references or len(references) == 0:
            hlog(f"Warning: Missing references for instance {request_state.instance}")
            return []

        # Check if the ground truth is answerable based on `is_impossible` flag
        ground_truth_query = references[0].output.text.strip() if references else None
        is_impossible = self.extract_is_impossible(input_text)

        is_answerable = not is_impossible and bool(ground_truth_query)  # True if the ground truth is answerable
        is_predicted_answerable = bool(prediction)  # True if the model generated a non-empty SQL query
        correct_answerable = int(is_answerable and is_predicted_answerable)  # Correct if both are answerable

        # **Execution Accuracy Fix:**
        execution_accuracy = 0

        if ground_truth_query:
            if ground_truth_result and predicted_result:
                execution_accuracy = int(set(predicted_result) == set(ground_truth_result))  # Compare sets.
            elif not ground_truth_result and not predicted_result and not prediction:
                execution_accuracy = 1  # Both empty and no query was generated.
        elif not ground_truth_query and prediction:
            execution_accuracy = 0  # LLM generated a query when no gold query exists.

        # **Query Validity Fix:**
        if not prediction:  # No SQL query was generated
            query_validity = 0
        elif query_error is None:
            query_validity = 1  # Query executed successfully.
        else:
            query_validity = 0  # Execution error occurred.

        return [
            # Execution-based Metrics
            Stat(MetricName("ehr_sql_execution_accuracy")).add(execution_accuracy),
            Stat(MetricName("ehr_sql_query_validity")).add(query_validity),

            # Answerability Metrics
            Stat(MetricName("ehr_sql_precision_answerable")).add(correct_answerable if is_predicted_answerable else 0),
            Stat(MetricName("ehr_sql_recall_answerable")).add(correct_answerable if is_answerable else 0),
            Stat(MetricName("ehr_sql_total_predicted_answerable")).add(int(is_predicted_answerable)),
            Stat(MetricName("ehr_sql_total_ground_truth_answerable")).add(int(is_answerable)),
        ]

    def compute(self, stats: List[Stat], **kwargs: Any) -> Dict[str, float]:
        """
        Aggregate statistics to compute final metrics.
        """

        execution_correct = sum(stat.value for stat in stats if stat.name == "ehr_sql_execution_accuracy")
        query_valid = sum(stat.value for stat in stats if stat.name == "ehr_sql_query_validity")

        correct_answerable = sum(
            stat.value for stat in stats if stat.name in ["ehr_sql_precision_answerable", "ehr_sql_recall_answerable"]
        )
        total_predicted_answerable = sum(
            stat.value for stat in stats if stat.name == "ehr_sql_total_predicted_answerable"
        )
        total_ground_truth_answerable = sum(
            stat.value for stat in stats if stat.name == "ehr_sql_total_ground_truth_answerable"
        )

        total_queries = len(stats) // 6  # Each query contributes 6 stats

        # Compute execution-based metrics
        execution_accuracy = execution_correct / total_queries if total_queries > 0 else 0.0
        query_validity = query_valid / total_queries if total_queries > 0 else 0.0

        # Compute answerability metrics
        precision = correct_answerable / total_predicted_answerable if total_predicted_answerable > 0 else 0.0
        recall = correct_answerable / total_ground_truth_answerable if total_ground_truth_answerable > 0 else 0.0

        return {
            "ehr_sql_execution_accuracy": execution_accuracy,
            "ehr_sql_query_validity": query_validity,
            "ehr_sql_precision_answerable": precision,
            "ehr_sql_recall_answerable": recall,
        }
