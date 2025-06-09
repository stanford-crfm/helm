from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.basic_metrics import BasicGenerationMetric
from helm.benchmark.metrics.metric import MetricSpec
from helm.common.object_spec import ObjectSpec

@run_spec_function("student_mistake_coding")
def get_student_mistake_coding_run_spec() -> RunSpec:
    return RunSpec(
        name="student_mistake_coding",
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.code_mistake_scenario.MistakeCodingScenario",
            args={}
        ),
        adapter_spec=AdapterSpec(
            method="generation",
            temperature=0.4,
            max_tokens=2000,
        ),
        metric_specs=[
            # Evaluation with AST, CodeBERT, and response alignment
            MetricSpec(
                class_name="helm.benchmark.metrics.code_evaluation_metrics.ComprehensiveCodeEvaluationMetric",
                args={"use_codebert": True, "compile_code": True}
            )
        ]
    )
