from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.basic_metrics import BasicGenerationMetric

@run_spec_function("student_coding")
def get_student_coding_run_spec() -> RunSpec:
    return RunSpec(
        name="student_coding",
        scenario_spec=ScenarioSpec(
            class_name="helm.benchmark.scenarios.student_coding_scenario.StudentCodingScenario",
            args={}
        ),
        adapter_spec=AdapterSpec(
            method="generation",
            temperature=0.4,
            max_tokens=2000,
        ),
        metric_specs=[BasicGenerationMetric(names=["exact_match","prefix_exact_match"])]
    )