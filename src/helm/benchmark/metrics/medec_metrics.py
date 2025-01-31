from typing import List, Dict, Any
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog
import re


class MedecMetric(Metric):
    """
    Metric for evaluating the MEDEC dataset, assessing error detection and correction.

    This implementation calculates two main metrics:
    1. Error Flag Accuracy: Whether the model correctly identifies if a clinical note contains an error.
    2. Error Sentence Detection Accuracy: Whether the model correctly identifies the erroneous sentence.
    """

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

        # Default values
        ground_truth_flag = 0  # Assume no error unless we find otherwise
        ground_truth_sentence_id = "-1"

        if references and len(references) > 0:
            ground_truth_reference = next(
                (ref for ref in references if "CORRECT_TAG" in ref.tags), None
            )

            if ground_truth_reference:
                # If the reference is "CORRECT", set flag = 0
                if ground_truth_reference.output.text == "CORRECT":
                    ground_truth_flag = 0
                    ground_truth_sentence_id = "-1"
                else:
                    # Otherwise, extract the error sentence ID
                    match = re.match(r"(\d+)", ground_truth_reference.output.text)
                    if match:
                        ground_truth_flag = 1
                        ground_truth_sentence_id = match.group(1)

        # Process prediction for correctness
        if prediction.startswith("CORRECT"):
            predicted_flag = 0
            predicted_sentence_id = "-1"
        else:
            match = re.match(r"(\d+)", prediction)
            predicted_sentence_id = match.group(1) if match else "-1"
            predicted_flag = 1 if predicted_sentence_id != "-1" else 0

        # Calculate accuracy
        flag_accuracy = int(predicted_flag == ground_truth_flag)
        sentence_accuracy = int(predicted_sentence_id == ground_truth_sentence_id)

        return [
            Stat(MetricName("medec_error_flag_accuracy")).add(flag_accuracy),
            Stat(MetricName("medec_error_sentence_accuracy")).add(sentence_accuracy),
        ]

    def compute(self, stats: List[Stat], **kwargs: Any) -> Dict[str, float]:
        """
        Aggregate statistics to compute final metrics.
        """
        total_flag_accuracy = sum(stat.value for stat in stats if stat.name == "medec_error_flag_accuracy")
        total_sentence_accuracy = sum(stat.value for stat in stats if stat.name == "medec_error_sentence_accuracy")

        count = len(stats) // 2  # Each instance contributes two stats
        return {
            "medec_error_flag_accuracy": total_flag_accuracy / count if count > 0 else 0.0,
            "medec_error_sentence_accuracy": total_sentence_accuracy / count if count > 0 else 0.0,
        }
