from typing import List
from helm.benchmark.metrics.metric import MetricSpec


def get_functional_correctness_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.codeinsights_correct_code_metrics.CodeInsightsFunctionalCorrectnessMetric",  # noqa: E501
            args={"timeout": 10, "max_workers": 1},
        )
    ]


def get_comprehensive_code_evaluation_metric_specs(use_codebert: bool = True) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.codeinsights_code_evaluation_metrics.CodeInsightsComprehensiveCodeEvaluationMetric",  # noqa: E501
            args={"use_codebert": use_codebert},
        )
    ]


def get_code_efficiency_metric_specs(
    num_runtime_runs: int = 5,
    timeout_seconds: int = 10,
    use_codebert: bool = True,  # ➊ add arg if you wish
):
    return [
        MetricSpec(  # existing metric → runtime & correctness
            class_name="helm.benchmark.metrics.codeinsights_code_efficiency_metrics.CodeInsightsCodeEfficiencyMetric",
            args={
                "num_runtime_runs": num_runtime_runs,
                "timeout_seconds": timeout_seconds,
            },
        ),
        MetricSpec(  # ➋ NEW metric → AST + CodeBERT
            class_name="helm.benchmark.metrics.codeinsights_code_evaluation_metrics.CodeInsightsCodeEvaluationMetric",
            args={"use_codebert": use_codebert},
        ),
    ]


def get_edge_case_metric_specs(
    use_codebert: bool = True,
) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.codeinsights_edge_case_metrics.CodeInsightsUnittestAlignmentMetric",  # noqa: E501
            args={"use_codebert": use_codebert},
        )
    ]
