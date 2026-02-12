"""Run spec functions for the HELM Reasoning scenarios."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _math_reasoning_adapter_spec() -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="Please reason step by step, and put your final answer within \\boxed{}.",
        input_prefix="",
        output_prefix="",
        max_tokens=4096,
        num_outputs=1,
        temperature=0.0,
    )


def _math_metric_spec() -> MetricSpec:
    return MetricSpec(class_name="helm.benchmark.metrics.math_metrics.MathVerifyMetric")


@run_spec_function("math500")
def get_math500_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.math500_scenario.Math500Scenario")
    adapter_spec = _math_reasoning_adapter_spec()
    metric_specs = get_basic_metric_specs([]) + [_math_metric_spec()]

    return RunSpec(
        name="math500",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["math500"],
    )


@run_spec_function("aime")
def get_aime_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.aime_scenario.AIMEScenario")
    adapter_spec = _math_reasoning_adapter_spec()
    metric_specs = get_basic_metric_specs([]) + [_math_metric_spec()]

    return RunSpec(
        name="aime",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["aime"],
    )


@run_spec_function("amc23")
def get_amc23_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.aime_scenario.AMC23Scenario")
    adapter_spec = _math_reasoning_adapter_spec()
    metric_specs = get_basic_metric_specs([]) + [_math_metric_spec()]

    return RunSpec(
        name="amc23",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["amc23"],
    )


@run_spec_function("aime25")
def get_aime25_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.aime_scenario.AIME25Scenario")
    adapter_spec = _math_reasoning_adapter_spec()
    metric_specs = get_basic_metric_specs([]) + [_math_metric_spec()]

    return RunSpec(
        name="aime25",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["aime25"],
    )

