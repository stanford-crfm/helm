from helm.benchmark.annotation.medication_qa_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MedicationQAMetric(LLMJuryMetric):
    """Score metrics for MedicationQA."""

    def __init__(self):
        super().__init__(
            metric_name="medication_qa_accuracy",
            scenario_name="medication_qa",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
