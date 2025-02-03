from typing import List, Dict, Any
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog
import re

class MIMICIVBillingCodeMetric(Metric):
    """
    Metric for evaluating the MIMIC Billing Code dataset, assessing the model's ability to match the
    reference ICD codes. In many cases the raw prediction output contains additional text (e.g., headers,
    bullet points, descriptions, markdown formatting) that make an exact string comparison unfair.
    
    This implementation first extracts ICD codes from both the reference(s) and the model prediction and then
    calculates:
    
      1. Precision for predicted ICD codes: the proportion of correctly predicted ICD codes among all predicted codes.
      2. Recall for predicted ICD codes: the proportion of correctly predicted ICD codes among all reference codes.
      3. F1 score: the harmonic mean of precision and recall.
    
    ICD codes are expected to have a letter followed by 1-3 digits, an optional period, and optional additional digits.
    For example: "J18.9", "J45.909", "J47.1", "J96.01", etc.
    """

    def extract_icd_codes(self, text: str) -> List[str]:
        """
        Extract ICD codes from a given text.
        
        This function removes markdown bold markers (e.g. **B20**) and then uses a regex pattern to
        capture strings that look like ICD codes. Any trailing period is removed from the captured code.
        """
        # Remove markdown bold formatting (or any double asterisks)
        cleaned_text = re.sub(r"\*\*", "", text)
        # The regex looks for a capital letter followed by 1-3 digits, optionally followed by a period and 1-4 digits.
        # It also allows for an optional trailing period.
        pattern = r"\b[A-Z]\d{1,3}(?:\.\d{1,4})?\.?\b"
        codes = re.findall(pattern, cleaned_text)
        # Remove any trailing period from each code (e.g., turning "Z21." into "Z21")
        codes = [code.rstrip(".") for code in codes]
        return codes

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        """
        For each instance, extract the ICD codes from both the gold reference(s) and the model's predicted text.
        Then compute the micro-average precision, recall, and F1 score over the entire dataset.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        from sklearn.preprocessing import MultiLabelBinarizer

        y_true = []  # list of lists of gold ICD codes for each instance
        y_pred = []  # list of lists of predicted ICD codes for each instance

        for request_state in request_states:
            # --- Extract ICD codes from reference(s) ---
            ref_codes = []
            for ref in request_state.instance.all_correct_references:
                if ref.output.text:
                    # Either the gold text is a comma-separated list or (to be safe) run it through the extractor.
                    ref_codes.extend(self.extract_icd_codes(ref.output.text))
            # Remove duplicates and extra whitespace
            ref_codes = list({code.strip() for code in ref_codes if code.strip()})
            y_true.append(ref_codes)

            # --- Extract ICD codes from the prediction ---
            if request_state.result is None or len(request_state.result.completions) == 0:
                pred_codes = []
            else:
                pred_text = request_state.result.completions[0].text
                pred_codes = self.extract_icd_codes(pred_text)
                pred_codes = list({code.strip() for code in pred_codes if code.strip()})
            y_pred.append(pred_codes)

        # --- Binarize the multi-label predictions ---
        # Gather all unique ICD codes seen in the references and predictions.
        all_labels = set()
        for codes in y_true:
            all_labels.update(codes)
        for codes in y_pred:
            all_labels.update(codes)
        all_labels = sorted(list(all_labels))

        # Initialize the multi-label binarizer with our set of ICD codes.
        mlb = MultiLabelBinarizer(classes=all_labels)
        # Note: with the 'classes' parameter provided, transform() will use these classes (order is preserved)
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        # --- Compute micro-average precision, recall, and F1 score ---
        precision = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

        hlog(f"MIMICBillingCodeMetric - Precision: {precision}, Recall: {recall}, F1: {f1}")

        return [
            Stat(MetricName("mimiciv_billing_code_precision")).add(precision),
            Stat(MetricName("mimiciv_billing_code_recall")).add(recall),
            Stat(MetricName("mimiciv_billing_code_f1")).add(f1),
        ]
