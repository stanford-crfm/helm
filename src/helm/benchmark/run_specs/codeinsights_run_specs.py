from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.codeinsights_metric_specs import (
    get_functional_correctness_metric_specs,
    get_comprehensive_code_evaluation_metric_specs,
    get_edge_case_metric_specs,
    get_code_efficiency_metric_specs,
)


@run_spec_function("correct_code")
def get_correct_code_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_correct_code_scenario.CorrectCodeScenario", args={}
    )

    instruction = (
        "You are a skilled C++ programmer working on a foundational programming course assignment. "
        "Your task is to write correct, efficient C++ code that solves the given problem. "
        "Write clean, well-structured code that follows good programming practices. "
        "Respond ONLY with the C++ code implementation (no commentary or explanations).\n\n"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        output_noun="Your code",
        stop_sequences=[],
        max_tokens=4000,
    )

    return RunSpec(
        name="correct_code",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_functional_correctness_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "correct_code"],
    )


@run_spec_function("student_coding")
def get_student_coding_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_student_coding_scenario.StudentCodingScenario", args={}
    )

    instruction = (
        "You are the same student who wrote the three examples below in your foundational C++ course. "
        "Mimic exactly your personal coding style, conventions, and level of proficiency—"
        "do not over‐optimize or introduce unfamiliar patterns. "
        "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
        "Respond ONLY with the C++ code (no commentary).\n\n"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction, output_noun="Your code", stop_sequences=[], max_tokens=4000
    )

    return RunSpec(
        name="student_coding",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_comprehensive_code_evaluation_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "student_coding"],
    )


@run_spec_function("student_mistake")
def get_student_mistake_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_student_mistake_scenario.StudentMistakeScenario", args={}
    )

    instruction = (
        "You are a C++ student with a consistent personal style, conventions, and proficiency level.\n"
        "Your task is to attempt the target problem **but introduce realistic mistake** you would typically make—"
        "do **not** provide a correct solution.\n\n"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction, output_noun="Your code", stop_sequences=[], max_tokens=4000
    )

    return RunSpec(
        name="student_mistake",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_comprehensive_code_evaluation_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "student_mistake"],
    )


@run_spec_function("code_efficiency")
def get_code_efficiency_run_spec() -> RunSpec:
    """
    Run specification for code efficiency evaluation scenario.

    This scenario evaluates whether LLM-generated code has similar runtime efficiency
    as the original student code. It focuses on problems where both solutions are
    functionally correct and measures runtime performance alignment.

    Requires C++ compiler (g++) to be available for actual compilation and execution.
    """
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_code_efficiency_scenario.CodeEfficiencyScenario", args={}
    )

    instruction = (
        "You are the same student who wrote the three examples below in your foundational C++ course. "
        "Mimic exactly your personal coding style, conventions, and make sure to generate a correct code. "
        "Do not over-optimize or introduce unfamiliar patterns. If the code is correct but inefficient, "
        "imitate the inefficiency. "
        "If the student writes efficiently, write efficiently too. "
        "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
        "Respond ONLY with the C++ code (no commentary).\n\n"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction, output_noun="Your code", stop_sequences=[], max_tokens=4000
    )

    return RunSpec(
        name="code_efficiency",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_code_efficiency_metric_specs(
            compile_code=True,  # Always compile for actual runtime measurement
            num_runtime_runs=5,  # Run each solution 5 times for averaging
            timeout_seconds=10,  # 10 second timeout per execution
        )
        + get_basic_metric_specs([]),
        groups=["codeinsights", "code_efficiency"],
    )


@run_spec_function("edge_case")
def get_edge_case_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_edge_case_scenario.EdgeCaseScenario", args={}
    )

    instruction = (
        "You are a student studying C++ with a consistent personal style, conventions, and proficiency level.\n"
        "Your task is to attempt the target problem **but make mistake in one of the test case—"
        "do **not** provide a complete correct solution.\n\n"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction, output_noun="Your code", stop_sequences=[], max_tokens=4000
    )

    return RunSpec(
        name="edge_case",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_edge_case_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "edge_case"],
    )
