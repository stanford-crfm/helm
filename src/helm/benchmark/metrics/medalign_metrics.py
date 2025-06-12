from helm.benchmark.annotation.medalign_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MedalignMetric(LLMJuryMetric):
    """Score metrics for Medalign."""

    def __init__(self):
        super().__init__(
            metric_name="medalign_accuracy",
            scenario_name="medalign",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
