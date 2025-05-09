from helm.benchmark.annotation.aci_bench_annotator import ANNOTATOR_MODELS
from helm.benchmark.metrics.llm_jury_metrics import LLMJuryMetric


class ACIBenchMetric(LLMJuryMetric):
    """Score metrics for ACIBench."""

    def __init__(self):
        super().__init__(
            metric_name="aci_bench_accuracy",
            scenario_name="aci_bench",
            annotator_models=ANNOTATOR_MODELS,
            default_score=1.0,
        )
