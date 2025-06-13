from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.codeinsights_metric_specs import get_comprehensive_code_evaluation_metric_specs
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs


@run_spec_function("student_mistake_coding")
def get_student_mistake_coding_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.code_mistake_scenario.MistakeCodingScenario", args={}
    )

    instruction = (
        "You are a C++ student with a consistent personal style, conventions, and proficiency level.\n"
        "Your task is to attempt the target problem **but introduce realistic mistake** you would typically make—"
        "do **not** provide a correct solution.\n\n"
    )

    adapter_spec = get_generation_adapter_spec(instructions=instruction, output_noun="Your code", max_tokens=4096)

    return RunSpec(
        name="student_mistake_coding",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_comprehensive_code_evaluation_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "student_mistake_coding"],
    )
