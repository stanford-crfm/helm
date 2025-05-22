from helm.benchmark.annotation.med_dialog_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MedDialogMetric(LLMJuryMetric):
    """Score metrics for MedDialog."""

    def __init__(self):
        super().__init__(
            metric_name="med_dialog_accuracy",
            scenario_name="med_dialog",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
