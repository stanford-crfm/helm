from helm.benchmark.annotation.mental_health_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MentalHealthMetric(LLMJuryMetric):
    """Score metrics for MentalHealth."""

    def __init__(self):
        super().__init__(
            metric_name="mental_health_accuracy",
            scenario_name="mental_health",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
