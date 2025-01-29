from typing import List, Dict, Any
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog
import re


class EhrSqlMetric(Metric):
    """
    Metric for evaluating the EHR SQL dataset, assessing the model's ability to generate valid SQL queries.

    This implementation calculates two main metrics:
    1. Precision for Answerable Questions (Pans): The proportion of correctly predicted answerable questions
       among all questions predicted to be answerable.
    2. Recall for Answerable Questions (Rans): The proportion of correctly predicted answerable questions
       among all answerable questions in the dataset.
    """

    def extract_value_from_text(self, input_text: str) -> bool:
        """Extract the value field from input_text instead of extra_data."""
        match = re.search(r"Is Impossible: (True|False)", input_text)
        return match and match.group(1) == "False"

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate a single generation against the reference labels.
        """

        # Extract predictions
        predictions = [
            completion.text.strip() for completion in request_state.result.completions
        ]

        if not predictions:
            raise ValueError("No predictions found in the completions.")

        # Process the first prediction as the primary output
        prediction = predictions[0]

        # Extract references and ground truth from the instance
        references = getattr(request_state.instance, "references", None)
        input_text = request_state.instance.input.text  # Read input text instead of extra_data

        if not references or len(references) == 0:
            hlog(f"Warning: Missing references for instance {request_state.instance}")
            return []

        # Check if the ground truth is answerable
        ground_truth_query = references[0].output.text
        is_answerable = bool(ground_truth_query) and self.extract_value_from_text(input_text)

        # Check if the model prediction is answerable
        is_predicted_answerable = bool(prediction)

        # Determine correctness for answerable questions
        correct_answerable = int(is_answerable and is_predicted_answerable)

        return [
            Stat(MetricName("ehr_sql_precision_answerable")).add(
                correct_answerable if is_predicted_answerable else 0
            ),
            Stat(MetricName("ehr_sql_recall_answerable")).add(
                correct_answerable if is_answerable else 0
            ),
            Stat(MetricName("ehr_sql_total_predicted_answerable")).add(
                int(is_predicted_answerable)
            ),
            Stat(MetricName("ehr_sql_total_ground_truth_answerable")).add(
                int(is_answerable)
            ),
        ]

    def compute(self, stats: List[Stat], **kwargs: Any) -> Dict[str, float]:
        """
        Aggregate statistics to compute final metrics.
        """

        # Sum up all relevant stats
        correct_answerable = sum(
            stat.value for stat in stats if stat.name in ["ehr_sql_precision_answerable", "ehr_sql_recall_answerable"]
        )
        total_predicted_answerable = sum(
            stat.value for stat in stats if stat.name == "ehr_sql_total_predicted_answerable"
        )
        total_ground_truth_answerable = sum(
            stat.value for stat in stats if stat.name == "ehr_sql_total_ground_truth_answerable"
        )

        # Calculate precision and recall
        precision = (
            correct_answerable / total_predicted_answerable
            if total_predicted_answerable > 0
            else 0.0
        )
        recall = (
            correct_answerable / total_ground_truth_answerable
            if total_ground_truth_answerable > 0
            else 0.0
        )

        return {
            "ehr_sql_precision_answerable": precision,
            "ehr_sql_recall_answerable": recall,
        }
