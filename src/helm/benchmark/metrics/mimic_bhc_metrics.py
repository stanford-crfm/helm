from helm.benchmark.annotation.mimic_bhc_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MIMICBHCMetric(LLMJuryMetric):
    """Score metrics for MIMICBHC."""

    def __init__(self):
        super().__init__(
            metric_name="mimic_bhc_accuracy",
            scenario_name="mimic_bhc",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
