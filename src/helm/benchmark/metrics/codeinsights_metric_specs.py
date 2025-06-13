from typing import List
from helm.benchmark.metrics.metric import MetricSpec


def get_functional_correctness_metric_specs(compile_code: bool = True) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.correct_code_metrics.FunctionalCorrectnessMetric",
            args={"compile_code": compile_code},
        )
    ]


def get_comprehensive_code_evaluation_metric_specs(
    use_codebert: bool = True, compile_code: bool = True
) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.code_evaluation_metrics.ComprehensiveCodeEvaluationMetric",
            args={"use_codebert": use_codebert, "compile_code": compile_code},
        )
    ]
