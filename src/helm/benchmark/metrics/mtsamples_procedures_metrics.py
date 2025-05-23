from helm.benchmark.annotation.mtsamples_procedures_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MTSamplesProceduresMetric(LLMJuryMetric):
    """Score metrics for MTSamplesProcedures."""

    def __init__(self):
        super().__init__(
            metric_name="mtsamples_procedures_accuracy",
            scenario_name="mtsamples_procedures",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
