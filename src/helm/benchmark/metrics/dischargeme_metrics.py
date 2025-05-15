from helm.benchmark.annotation.dischargeme_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class DischargeMeMetric(LLMJuryMetric):
    """Score metrics for DischargeMe."""

    def __init__(self):
        super().__init__(
            metric_name="dischargeme_accuracy",
            scenario_name="dischargeme",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
