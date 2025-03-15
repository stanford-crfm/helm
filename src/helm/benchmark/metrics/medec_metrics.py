from typing import List
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog
import re
from helm.benchmark.scenarios.scenario import CORRECT_TAG


class MedecMetric(Metric):
    """
    Metric for evaluating the MEDEC dataset, assessing medical error detection and correction.

    - **Error Flag Accuracy**: Whether the model correctly identifies if a medical note contains an error.
    - **Error Sentence Detection Accuracy**: Whether the model correctly identifies the erroneous
        sentence when an error is present.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate a single LLM generation against the ground truth labels.
        """

        # Extract predictions from the model output
        if request_state.result is not None:
            predictions = [completion.text.strip() for completion in request_state.result.completions]
        else:
            predictions = []

        if not predictions:
            raise ValueError("No predictions found in the completions.")

        prediction = predictions[0]  # Use the first completion

        # Extract references (ground truth)
        references = getattr(request_state.instance, "references", None)

        if not references or len(references) == 0:
            hlog(f"Warning: Missing references for instance {request_state.instance}")
            return []

        ground_truth_reference = next((ref for ref in references if CORRECT_TAG in ref.tags), None)

        if not ground_truth_reference:
            hlog(f"Warning: No ground truth reference with CORRECT_TAG for instance {request_state.instance}")
            return []

        # Extract the ground truth error flag and sentence ID
        # ground_truth_text_pre = ground_truth_reference.output.text.strip()
        ground_truth_text = ground_truth_reference.output.text.replace(".", "").strip()

        # Determine ground truth values
        if ground_truth_text == "CORRECT":
            ground_truth_flag = 0  # No medical error
            ground_truth_sentence_id = -1  # No sentence should be flagged
        else:
            ground_truth_flag = 1  # Medical error exists
            try:
                ground_truth_sentence_id = int(ground_truth_text)
            except ValueError:
                ground_truth_sentence_id = -1

        # Determine model predictions
        prediction = prediction.strip()

        if prediction == "CORRECT":
            predicted_flag = 0
            predicted_sentence_id = -1
        else:
            match = re.match(r"^(\d+)", prediction)  # Ensure first token is a number
            if match:
                predicted_sentence_id = int(match.group(1))
                predicted_flag = 1  # A valid sentence ID implies the model thinks there's an error
            else:
                predicted_flag = 1  # The model attempted to correct something, but in an invalid format
                predicted_sentence_id = -2  # Mark invalid outputs

        # Calculate Error Flag Accuracy
        flag_accuracy = 1 if predicted_flag == ground_truth_flag else 0

        # Calculate Error Sentence Accuracy
        if ground_truth_flag == 1 and predicted_flag == 1:
            sentence_accuracy = 1 if predicted_sentence_id == ground_truth_sentence_id else 0
        else:
            sentence_accuracy = (
                1 if ground_truth_flag == 0 and predicted_flag == 0 else 0
            )  # Both must agree it's "CORRECT"

        return [
            Stat(MetricName("medec_error_flag_accuracy")).add(flag_accuracy),
            Stat(MetricName("medec_error_sentence_accuracy")).add(sentence_accuracy),
        ]
