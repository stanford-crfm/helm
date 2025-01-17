from typing import List, Dict, Any
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog

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
        # Debugging output
        #print(f"Current completions: {request_state.result.completions}")

        # Extract predictions
        predictions = [
            completion.text.strip() for completion in request_state.result.completions
        ]

        if not predictions:
            raise ValueError("No predictions found in the completions.")

        # Process the first prediction as the primary output
        prediction = predictions[0]

        # Try to get references from the instance
        references = getattr(request_state.instance, "references", None)
        if not references or len(references) == 0:
            # If references are missing, log and skip evaluation for this instance
            print(f"Warning: Missing references for instance {request_state.instance}")
            return []

        # The ground truth is the first reference with a CORRECT_TAG
        ground_truth_reference = next(
            (ref for ref in references if "CORRECT_TAG" in ref.tags), None
        )
        if not ground_truth_reference:
            print(f"Warning: No ground truth reference with CORRECT_TAG for instance {request_state.instance}")
            return []

        # Ground truth flag and sentence
        ground_truth_flag = 1 if ground_truth_reference.output.text else 0
        ground_truth_sentence = (
            int(ground_truth_reference.output.text.split()[0])
            if ground_truth_flag
            else -1
        )

        # Process prediction
        if prediction.startswith("CORRECT"):
            predicted_flag = 0
            predicted_sentence = -1
        else:
            split_prediction = prediction.split()
            if len(split_prediction) > 0 and split_prediction[0].isdigit():
                predicted_sentence = int(split_prediction[0])
                predicted_flag = 1 if predicted_sentence != -1 else 0
            else:
                predicted_sentence = -1
                predicted_flag = 0

        # Calculate metrics
        flag_accuracy = int(predicted_flag == ground_truth_flag)
        sentence_accuracy = int(predicted_sentence == ground_truth_sentence)

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
