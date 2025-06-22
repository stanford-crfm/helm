from typing import List
from helm.benchmark.metrics.metric import MetricSpec


def get_functional_correctness_metric_specs(compile_code: bool = True) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.codeinsights_correct_code_metrics.FunctionalCorrectnessMetric",
            args={"compile_code": compile_code},
        )
    ]


def get_comprehensive_code_evaluation_metric_specs(
    use_codebert: bool = True, compile_code: bool = True
) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.codeinsights_code_evaluation_metrics.ComprehensiveCodeEvaluationMetric",
            args={"use_codebert": use_codebert, "compile_code": compile_code},
        )
    ]


def get_code_efficiency_metric_specs(
    compile_code: bool = True,
    num_runtime_runs: int = 5,
    timeout_seconds: int = 10,
    use_codebert: bool = True,  # ➊ add arg if you wish
):
    return [
        MetricSpec(  # existing metric → runtime & correctness
            class_name="helm.benchmark.metrics.codeinsights_code_efficiency_metrics.CodeEfficiencyMetric",
            args={
                "compile_code": compile_code,
                "num_runtime_runs": num_runtime_runs,
                "timeout_seconds": timeout_seconds,
            },
        ),
        MetricSpec(  # ➋ NEW metric → AST + CodeBERT
            class_name="helm.benchmark.metrics.codeinsights_code_evaluation_metrics.CodeEvaluationMetric",
            args={"use_codebert": use_codebert},
        ),
    ]


def get_edge_case_metric_specs(
    use_codebert: bool = True,
    compile_code: bool = True,
    compiler_path: str = "g++",
) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.codeinsights_edge_case_metrics.ComprehensiveCodeEvaluationMetric",
            args={
                "use_codebert": use_codebert,
                "compile_code": compile_code,
                "compiler_path": compiler_path,
            },
        )
    ]
