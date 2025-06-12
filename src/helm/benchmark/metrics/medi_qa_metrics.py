from helm.benchmark.annotation.medi_qa_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MediQAMetric(LLMJuryMetric):
    """Score metrics for MediQA."""

    def __init__(self):
        super().__init__(
            metric_name="medi_qa_accuracy",
            scenario_name="medi_qa",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
