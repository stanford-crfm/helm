from helm.benchmark.annotation.starr_patient_instructions_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class StarrPatientInstructionsMetric(LLMJuryMetric):
    """Score metrics for StarrPatientInstructions."""

    def __init__(self):
        super().__init__(
            metric_name="starr_patient_instructions_accuracy",
            scenario_name="starr_patient_instructions",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
