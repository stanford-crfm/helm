from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.basic_metrics import BasicGenerationMetric
from helm.benchmark.metrics.metric import MetricSpec
from helm.common.object_spec import ObjectSpec

@run_spec_function("correct_code")
def get_student_coding_run_spec() -> RunSpec:
    return RunSpec(
        name="correct_code",
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.correct_code_scenario.CorrectCodeScenario",
            args={}
        ),
        adapter_spec=AdapterSpec(
            method="generation",
            temperature=0,
            max_tokens=2000,
        ),
        metric_specs=[
            # Evaluation with AST, CodeBERT, and response alignment
            MetricSpec(
                class_name="helm.benchmark.metrics.correct_code_metrics.FunctionalCorrectnessMetric",
                args={"compile_code": True}
            )
        ]
    )