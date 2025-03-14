from typing import List
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
            hlog(f"Warning: Request state missing annotations for instance {request_state.instance}")
            return []

        if "ehr_sql" not in request_state.annotations:
            hlog(f"Warning: 'ehr_sql' key missing in annotations for instance {request_state.instance}")
            return []

        # Extract execution results
        predicted_result = request_state.annotations["ehr_sql"].get("predicted_result", [])
        ground_truth_result = request_state.annotations["ehr_sql"].get("ground_truth_result", [])
        query_error = request_state.annotations["ehr_sql"].get("query_error", None)

        # Extract predictions from the model output
        if request_state.result is None:
            predictions = []
        else:
            predictions = [completion.text.strip() for completion in request_state.result.completions]
        if not predictions:
            hlog(f"Warning: No predictions found in the completions for instance {request_state.instance}")
            return []

        # Process the first prediction as the primary output
        prediction = predictions[0].strip()

        # Extract references and input text
        references = getattr(request_state.instance, "references", None)

        if not references or len(references) == 0:
            hlog(f"Warning: Missing references for instance {request_state.instance}")
            return []

        # Check if the ground truth is answerable based on `is_impossible` flag
        ground_truth_query = references[0].output.text.strip() if references else None
        is_impossible = (
            request_state.instance.extra_data.get("is_impossible", False)
            if request_state.instance.extra_data
            else False
        )

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
