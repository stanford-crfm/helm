from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.codeinsights_metric_specs import get_comprehensive_code_evaluation_metric_specs
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs


@run_spec_function("student_coding")
def get_student_coding_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.student_coding_scenario.StudentCodingScenario", args={}
    )

    instruction = (
        "You are the same student who wrote the three examples below in your foundational C++ course. "
        "Mimic exactly your personal coding style, conventions, and level of proficiency—"
        "do not over‐optimize or introduce unfamiliar patterns. "
        "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
        "Respond ONLY with the C++ code (no commentary).\n\n"
    )

    adapter_spec = get_generation_adapter_spec(instructions=instruction, output_noun="Your code", max_tokens=4096)

    return RunSpec(
        name="student_coding",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_comprehensive_code_evaluation_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "student_coding"],
    )
