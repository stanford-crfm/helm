from typing import List
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


class MIMICIVBillingCodeMetric(Metric):
    """
    Metric for evaluating the MIMIC Billing Code dataset, assessing the model's ability to match the
    reference ICD codes. Handles cases where raw prediction output contains additional text.

    Calculates:
    1. Precision: proportion of correctly predicted ICD codes among all predicted codes
    2. Recall: proportion of correctly predicted ICD codes among all reference codes
    3. F1 score: harmonic mean of precision and recall

    ICD codes format: letter followed by 1-3 digits, optional period, optional additional digits
    Examples: "J18.9", "J45.909", "J47.1", "J96.01"
    """

    def extract_icd_codes(self, text: str) -> List[str]:
        """Extract ICD codes from text, handling markdown and standardizing format."""
        if not text:
            return []

        # Remove markdown bold formatting
        cleaned_text = re.sub(r"\*\*", "", text)
        # Match ICD code pattern with optional period and trailing digits
        pattern = r"\b[A-Z]\d{1,3}(?:\.\d{1,4})?\.?\b"
        codes = re.findall(pattern, cleaned_text)
        # Standardize by removing trailing periods
        return [code.rstrip(".").strip() for code in codes]

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        """
        Evaluate a single generation against reference labels.
        """
        # Extract predictions
        if request_state.result is None:
            predictions = []
        else:
            predictions = [completion.text.strip() for completion in request_state.result.completions]
        if not predictions:
            hlog("Warning: No predictions found in completions")
            return []

        # Get the first prediction
        prediction = predictions[0]

        # Get references
        references = getattr(request_state.instance, "references", None)

        if not references or len(references) == 0:
            hlog(f"Warning: Missing references for instance {request_state.instance}")
            return []

        # Extract codes from reference and prediction
        ref_codes = []
        for ref in references:
            if ref.output.text:
                ref_codes.extend(self.extract_icd_codes(ref.output.text))
        ref_codes = list(set(ref_codes))  # Remove duplicates

        pred_codes = self.extract_icd_codes(prediction)
        pred_codes = list(set(pred_codes))  # Remove duplicates

        # Convert to binary format for metrics
        all_codes = sorted(list(set(ref_codes + pred_codes)))
        mlb = MultiLabelBinarizer(classes=all_codes)

        y_true_bin = mlb.fit_transform([ref_codes])
        y_pred_bin = mlb.transform([pred_codes])

        # Calculate metrics
        precision = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

        return [
            Stat(MetricName("mimiciv_billing_code_precision")).add(precision),
            Stat(MetricName("mimiciv_billing_code_recall")).add(recall),
            Stat(MetricName("mimiciv_billing_code_f1")).add(f1),
        ]
