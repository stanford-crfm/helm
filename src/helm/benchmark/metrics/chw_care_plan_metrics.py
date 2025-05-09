from helm.benchmark.annotation.chw_care_plan_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class CHWCarePlanMetric(LLMJuryMetric):
    """Score metrics for CHWCarePlan."""

    def __init__(self):
        super().__init__(
            metric_name="chw_care_plan_accuracy",
            scenario_name="chw_care_plan",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
