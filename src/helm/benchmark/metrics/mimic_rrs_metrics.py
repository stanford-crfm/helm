from helm.benchmark.annotation.mimic_rrs_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MIMICRRSMetric(LLMJuryMetric):
    """Score metrics for MIMICRRS."""

    def __init__(self):
        super().__init__(
            metric_name="mimic_rrs_accuracy",
            scenario_name="mimic_rrs",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
