from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("medcalc_bench")
def get_medcalc_bench_spec(method: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medcalc_bench_scenario.MedCalcBenchScenario",
        args={"is_one_shot": method == "one_shot"},
    )

    if method == "zero_shot":
        adapter_spec = get_medcalc_bench_adapter()
    elif method == "one_shot":
        adapter_spec = get_medcalc_bench_adapter()
    else:
        raise ValueError(f"Invalid method for MedCalc-Bench: {method}")

    metric_specs: List[MetricSpec] = [
        MetricSpec(
            class_name="helm.benchmark.metrics.medcalc_bench_metrics.MedCalcBenchMetric",
            args={},
        )
    ] # + get_basic_metric_specs([])

    return RunSpec(
        name=f"medcalc_bench:method={method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["medcalc_bench"],
    )


def get_medcalc_bench_adapter() -> AdapterSpec:
    return get_generation_adapter_spec(
        instructions=_get_cot_instructions(),
        input_noun=None,  # Set directly in the scenario.
        output_noun="\n\nCalculated Value",
        max_tokens=500,
    )

def _get_cot_instructions() -> str:
    """Generate instructions for the MedCalc-Bench scenario.

    This function is inspired on the system prompt definition in the original code:
    https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/run.py#L16

    Credits to the original authors: https://github.com/ncbi-nlp/MedCalc-Bench.
    """

    return (
        "You are a helpful assistant for calculating a score for a given patient note. "
        "Please think step-by-step to solve the question and then generate the required score. "
        "Your output should contain the step by step thinking and the final answer, which is a short and direct answer to the question. "
        'Before giving the final answer, write "Calculated Value: " followed by the answer.'
    )
