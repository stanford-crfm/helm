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


@run_spec_function("codeinsights_correct_code")
def get_codeinsights_correct_code_run_spec(tpr: float = 0.0, num_testcases: int = 1) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_correct_code_scenario.CodeInsightsCorrectCodeScenario",
        args={"num_testcases": num_testcases},
    )

    instruction = (
        "You are a skilled C++ programmer working on a foundational programming course assignment. "
        "Your task is to write correct, efficient C++ code that solves the given problem. "
        "Write clean, well-structured code that follows good programming practices. "
        "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template."
        "DO NOT reproduce the template part as the generated code would be inserted to the template,"
        "and make sure the code is compatible with the Unit Test Input"
        "Ensure your code is correct, efficient, includes any class definition when needed, and handles all edge cases properly."
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        output_noun="Your code",
        stop_sequences=[],
        max_tokens=4000,
        temperature=tpr,
    )

    return RunSpec(
        name=f"codeinsights_correct_code:temperature={adapter_spec.temperature},num_testcases={num_testcases}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_functional_correctness_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "codeinsights_correct_code"],
    )


@run_spec_function("codeinsights_student_coding")
def get_codeinsights_student_coding_run_spec(tpr: float = 0.0, num_testcases: int = 1) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_student_coding_scenario.CodeInsightsStudentCodingScenario",
        args={"num_testcases": num_testcases},
    )

    instruction = (
        "You are the same student who wrote the three examples below in your foundational C++ course. "
        "Mimic exactly your personal coding style, conventions, and level of proficiency—"
        "do not over‐optimize or introduce unfamiliar patterns. "
        "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
        "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template."
        "DO NOT reproduce the template part as the generated code would be inserted to the template,"
        "and make sure the code is compatible with the Unit Test Input"
        "Ensure your code includes any class definition when needed."
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        output_noun="Your code",
        stop_sequences=[],
        max_tokens=4000,
        temperature=tpr,
    )

    return RunSpec(
        name=f"codeinsights_student_coding:temperature={adapter_spec.temperature},num_testcases={num_testcases}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_comprehensive_code_evaluation_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "codeinsights_student_coding"],
    )


@run_spec_function("codeinsights_student_mistake")
def get_codeinsights_student_mistake_run_spec(tpr: float = 0.0, num_testcases: int = 1) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_student_mistake_scenario.CodeInsightsStudentMistakeScenario",
        args={"num_testcases": num_testcases},
    )

    instruction = (
        "You are a C++ student with a consistent personal style, conventions, and proficiency level.\n"
        "Your task is to attempt the target problem **but introduce realistic mistake** you would typically make—"
        "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template."
        "DO NOT reproduce the template part as the generated code would be inserted to the template,"
        "and make sure the code is compatible with the Unit Test Input"
        "Ensure your code is includes any class definition when needed."
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        output_noun="Your code",
        stop_sequences=[],
        max_tokens=4000,
        temperature=tpr,
    )

    return RunSpec(
        name=f"codeinsights_student_mistake:temperature={adapter_spec.temperature},num_testcases={num_testcases}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_comprehensive_code_evaluation_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "codeinsights_student_mistake"],
    )


@run_spec_function("codeinsights_code_efficiency")
def get_codeinsights_code_efficiency_run_spec(tpr: float = 0.0, num_testcases: int = 1) -> RunSpec:
    """
    Run specification for code efficiency evaluation scenario.

    This scenario evaluates whether LLM-generated code has similar runtime efficiency
    as the original student code. It focuses on problems where both solutions are
    functionally correct and measures runtime performance alignment.

    Requires C++ compiler (g++) to be available for actual compilation and execution.
    """
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_code_efficiency_scenario.CodeInsightsCodeEfficiencyScenario",
        args={"num_testcases": num_testcases},
    )

    instruction = (
        "You are the same student who wrote the three examples below in your foundational C++ course. "
        "Mimic exactly your personal coding style, conventions, and make sure to generate a correct code. "
        "Do not over-optimize or introduce unfamiliar patterns. If the code is correct but inefficient, "
        "imitate the inefficiency. "
        "If the student writes efficiently, write efficiently too. "
        "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
        "Provide ONLY your C++ implementation following the given template, where the answer will replace the {{ STUDENT_ANSWER }} block in the template."
        "DO NOT reproduce the template part as the generated code would be inserted to the template,"
        "and make sure the code is compatible with the Unit Test Input"
        "Ensure your code is correct, includes any class definition when needed, and handles all edge cases properly."
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        output_noun="Your code",
        stop_sequences=[],
        max_tokens=4000,
        temperature=tpr,
    )

    return RunSpec(
        name=f"codeinsights_code_efficiency:temperature={adapter_spec.temperature},num_testcases={num_testcases}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_code_efficiency_metric_specs(
            num_runtime_runs=5,  # Run each solution 5 times for averaging
            timeout_seconds=10,  # 10 second timeout per execution
        )
        + get_basic_metric_specs([]),
        groups=["codeinsights", "codeinsights_code_efficiency"],
    )


@run_spec_function("codeinsights_edge_case")
def get_codeinsights_edge_case_run_spec(tpr: float = 0.0, num_testcases: int = 1) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.codeinsights_edge_case_scenario.CodeInsightsEdgeCaseScenario",
        args={"num_testcases": num_testcases},
    )

    instruction = (
        "You are a student studying C++ with a consistent personal style, conventions, and proficiency level.\n"
        "Your task is to identify which test case you would likely to fail for a given question with unit tests.\n"
        "Respond only with integer of the unittest number\n\n"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction,
        output_noun="Your code",
        stop_sequences=[],
        max_tokens=4000,
        temperature=tpr,
    )

    return RunSpec(
        name=f"codeinsights_edge_case:temperature={adapter_spec.temperature},num_testcases={num_testcases}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_edge_case_metric_specs() + get_basic_metric_specs([]),
        groups=["codeinsights", "codeinsights_edge_case"],
    )
