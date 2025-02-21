from typing import Any, Dict, List
from helm.benchmark.metrics.metric import MetricSpec


def get_seahelm_machine_translation_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.seahelm_metrics.SEAHELMMachineTranslationMetric")]


def get_seahelm_qa_metric_specs(args: Dict[str, Any]) -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.seahelm_metrics.SEAHELMQAMetric", args=args)]
