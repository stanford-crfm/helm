from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.basic_metrics import BasicGenerationMetric
from helm.benchmark.metrics.metric import MetricSpec
from helm.common.object_spec import ObjectSpec

@run_spec_function("student_coding")
def get_student_coding_run_spec() -> RunSpec:
    return RunSpec(
        name="student_coding",
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.student_coding_scenario.StudentCodingScenario",
            args={}
        ),
        adapter_spec=AdapterSpec(
            method="chat_completions",
            temperature=0.4,
            max_tokens=2000,
        ),
        metric_specs=[
            # AST + CodeBERT evaluation metrics (recommended)
            MetricSpec(
                class_name="helm.benchmark.metrics.code_evaluation_metrics.CodeEvaluationMetric",
                args={"use_codebert": True}
            ),
            
            # Advanced code evaluation with quality metrics
            MetricSpec(
                class_name="helm.benchmark.metrics.code_evaluation_metrics.AdvancedCodeEvaluationMetric", 
                args={"use_codebert": True}
            )
        ]
    )

@run_spec_function("student_coding_ast_only")
def get_student_coding_ast_only_run_spec() -> RunSpec:
    """Alternative run spec using only AST metrics (faster, no model download required)."""
    return RunSpec(
        name="student_coding_ast_only",
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.student_coding_scenario.StudentCodingScenario",
            args={}
        ),
        adapter_spec=AdapterSpec(
            method="chat_completions",
            temperature=0.4,
            max_tokens=2000,
        ),
        metric_specs=[
            # AST-only evaluation metrics (no CodeBERT)
            MetricSpec(
                class_name="helm.benchmark.metrics.code_evaluation_metrics.CodeEvaluationMetric",
                args={"use_codebert": False}
            ),
            
            # Advanced AST-only evaluation
            MetricSpec(
                class_name="helm.benchmark.metrics.code_evaluation_metrics.AdvancedCodeEvaluationMetric", 
                args={"use_codebert": False}
            )
        ]
    )