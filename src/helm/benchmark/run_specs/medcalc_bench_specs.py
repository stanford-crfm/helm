import json
from typing import Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

ONE_SHOT_EXAMPLES_URL = "https://raw.githubusercontent.com/ncbi-nlp/MedCalc-Bench/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/one_shot_finalized_explanation.json"


@run_spec_function("medcalc_bench")
def get_medcalc_bench_spec(method: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medcalc_bench_scenario.MedCalcBenchScenario",
        args={},
    )

    if method == "zero_shot":
        adapter_spec = get_medcalc_bench_zero_shot_adapter()
    elif method == "one_shot":
        adapter_spec = get_medcalc_bench_one_shot_adapter()
    else:
        raise ValueError(f"Invalid method for MedCalc-Bench: {method}")

    metric_specs: List[MetricSpec] = [
        MetricSpec(
            class_name="helm.benchmark.metrics.medcalc_bench_metrics.MedCalcBenchMetric",
            args={},
        )
    ] # + get_basic_metric_specs([])

    return RunSpec(
        name=f"medcalc_bench:method{method}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["medcalc_bench"],
    )


def get_medcalc_bench_zero_shot_adapter() -> AdapterSpec:
    return get_generation_adapter_spec(
        instructions=_get_zero_shot_cot_instructions(),
        input_noun=None,  # Set directly in the scenario.
        output_noun="\n\nCalculated Value",
        max_tokens=500,
    )


def get_medcalc_bench_one_shot_adapter() -> AdapterSpec:
    return get_generation_adapter_spec(
        instructions=_get_one_shot_cot_instructions(
            # TODO: Modify this to retrieve the question and calculator ID from the respective dataset sample.
            # For more information see the docstring for the `_get_one_shot_cot_instructions` function.
            # One way of doing so is having receiving the calculator ID in this function and passing it to
            # the scenario, which can then filter the dataset samples by the calculator ID.
            question=(
                "What is the patient's Creatinine Clearance using the Cockroft-Gault Equation in terms of mL/min? "
                "You should use the patient's adjusted body weight in kg instead of the patient's actual body "
                "weight if the patient is overweight or obese based on their BMI. If the patient's BMI's normal, "
                "set their adjusted body weight to the minimum of the ideal body and actual weight. If the "
                "patient is underweight, please set their adjusted body weight to their actual body weight."
            ),
            calculator_id="2",
        ),
        input_noun=None,  # Set directly in the scenario.
        output_noun="\n\nCalculated Value",
        max_tokens=500,
    )


def _get_zero_shot_cot_instructions() -> str:
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


def _get_one_shot_cot_instructions(question: str, calculator_id: str) -> str:
    """Generate instructions for the MedCalc-Bench scenario.

    This function is inspired on the system prompt definition in the original code:
    https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/run.py#L26

    Credits to the original authors: https://github.com/ncbi-nlp/MedCalc-Bench.

    In the original code, there's exactly one example response for each calculator ID.
    These examples are stored in a JSON file: https://github.com/ncbi-nlp/MedCalc-Bench/blob/048ba77dbe332e9190935e4a30965bff444b940e/evaluation/one_shot_finalized_explanation.json
    None of the examples include the actual questions. They only contain the step-by-step thinking and the final answer.
    Looking at the dataset samples we can see that all samples with the same calculator ID use the same question.
    The original expect that for each sample, we collect the calculator ID and the question for building the one-shot instructions.
    """
    examples: Dict = {}
    with open(ONE_SHOT_EXAMPLES_URL, "r") as f:
        examples = json.load(f)

    if not examples:
        raise ValueError(
            "Failed to load one-shot examples for the MedCalc-Bench scenario."
        )

    example = examples.get(calculator_id, {})

    if not example:
        raise ValueError(
            f"Failed to find one-shot example for calculator ID {calculator_id}."
        )

    return (
        "You are a helpful assistant for calculating a score for a given patient note. "
        "Please think step-by-step to solve the question and then generate the required score. "
        "Your output should contain the step by step thinking and the final answer, which is a short and direct answer to the question. "
        "\nBelow is an example:"
        # This example follows the formatting of the respective scenario.
        f"Patient Note:\n\n{example['Patient Note']}"
        f"\n\nQuestion:\n\n{question}"
        f"\n\nExplanation:\n\n{example['Response']['step_by_step_thinking']}"
        f"\n\nCalculated Value: {example['Response']['answer']}"
    )
