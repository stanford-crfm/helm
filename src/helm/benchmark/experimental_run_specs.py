"""Run specs for experiments only.

These run specs are not intended for use with public leaderboards."""

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
