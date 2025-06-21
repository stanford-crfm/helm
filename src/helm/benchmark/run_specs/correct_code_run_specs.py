from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.codeinsights_metric_specs import get_functional_correctness_metric_specs
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs


@run_spec_function("correct_code")
def get_student_coding_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.correct_code_scenario.CorrectCodeScenario", args={}
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
