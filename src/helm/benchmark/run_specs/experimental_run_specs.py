"""Run specs for experiments only.

These run specs are not intended for use with public leaderboards."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ci_mcqa")
def get_ci_mcqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.ci_mcqa_scenario.CIMCQAScenario", args={})

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=(
            "Give a letter answer among the options given. "
            "For example, if the options are A, B, C, D, E, and F, "
            "your answer should consist of the single letter that corresponds to the correct answer."
        ),
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="ci_mcqa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["CIMCQA"],
    )


@run_spec_function("ewok")
def get_ewok_spec(domain: str = "all") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ewok_scenario.EWoKScenario", args={"domain": domain}
    )

    instructions = """# INSTRUCTIONS

In this study, you will see multiple examples. In each example, you will be given two contexts and a scenario. Your task is to read the two contexts and the subsequent scenario, and pick the context that makes more sense considering the scenario that follows. The contexts will be numbered "1" or "2". You must answer using "1" or "2" in your response.
"""  # noqa: E501
    input_prefix = """# TEST EXAMPLE

## Scenario
\""""
    input_suffix = """\"

## Contexts
"""
    output_prefix = """
## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
"""

    adapter_spec = AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions=instructions,
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        reference_prefix='1. "',
        reference_suffix='"\n',
        output_prefix=output_prefix,
        output_suffix="\n",
        max_train_instances=2,
        num_outputs=1,
        max_tokens=2,
        temperature=0.0,
        stop_sequences=["\n\n"],
        sample_train=True,
    )

    return RunSpec(
        name=f"ewok:domain={domain}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["ewok", f"ewok_{domain}"],
    )
