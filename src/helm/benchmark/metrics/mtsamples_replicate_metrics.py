from helm.benchmark.annotation.mtsamples_replicate_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class MTSamplesReplicateMetric(LLMJuryMetric):
    """Score metrics for MTSamplesReplicate."""

    def __init__(self):
        super().__init__(
            metric_name="mtsamples_replicate_accuracy",
            scenario_name="mtsamples_replicate",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
