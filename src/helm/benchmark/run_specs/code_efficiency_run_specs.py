from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.codeinsights_metric_specs import get_code_efficiency_metric_specs
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs


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
        class_name="helm.benchmark.scenarios.efficiency_alignment_scenario.CodeEfficiencyScenario", 
        args={}
    )

    instruction = (
        "You are the same student who wrote the three examples below in your foundational C++ course. "
        "Mimic exactly your personal coding style, conventions, and make sure to generate a correct code. "
        "Do not over-optimize or introduce unfamiliar patterns. If the code is correct but inefficient, imitate the inefficiency. "
        "If the student writes efficiently, write efficiently too. "
        "Include the same sort of formatting, variable names, and minor imperfections you demonstrated. "
        "Respond ONLY with the C++ code (no commentary).\n\n"
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=instruction, 
        output_noun="Your code", 
        max_tokens=4000
    )

    return RunSpec(
        name="code_efficiency",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_code_efficiency_metric_specs(
            compile_code=True,      # Always compile for actual runtime measurement
            num_runtime_runs=5,     # Run each solution 5 times for averaging
            timeout_seconds=10      # 10 second timeout per execution
        ) + get_basic_metric_specs([]),
        groups=["codeinsights", "code_efficiency"],
    )