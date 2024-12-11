from typing import List

from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("med_calc_bench_zero_shot_cot")
def get_med_calc_bench_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.med_calc_bench_scenario.MedCalcBenchScenario",
        args={},
    )

    adapter_spec = get_generation_adapter_spec(
        instructions=_get_zero_shot_cot_instructions(),
        input_noun="Patient Note",
        output_noun="Calculated Value",
        max_tokens=50,
    )

    metric_specs: List[MetricSpec] = [
        MetricSpec(
            class_name="helm.benchmark.metrics.med_calc_bench_metrics.MedCalcBenchMetric",
            args={},
        )
    ] + get_basic_metric_specs([])

    return RunSpec(
        name="med_calc_bench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["med_calc_bench"],
    )


def _get_zero_shot_cot_instructions() -> str:
    """Generate instructions for the MedCalcBench scenario.

    This function is inspired on the system prompt definition in the original code:
    https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/run.py#L16

    Credits to the original authors: https://github.com/ncbi-nlp/MedCalc-Bench.
    """

    return (
        "You are a helpful assistant for calculating a score for a given patient note. "
        "Please think step-by-step to solve the question and then generate the required score. "
        "Your output should contain the step by step thinking and the final answer, which is a short and direct answer to the question. "
        'Before giving the final answer, write "Final Answer: " followed by the answer.'
    )
