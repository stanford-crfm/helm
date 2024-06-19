from typing import Any, Dict, List
from helm.benchmark.metrics.metric import MetricSpec


def get_bhasa_machine_translation_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.bhasa_metrics.BhasaMachineTranslationMetric")]


def get_bhasa_qa_metric_specs(args: Dict[str, Any]) -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.bhasa_metrics.BhasaQAMetric", args=args)]
