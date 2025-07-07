"""Run specs for Arabic leaderboard

EXPERIMENTAL: Run specs here may have future reverse incompatible changes."""

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_adapter_spec, get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("arabic_mmlu")
def get_arabic_mmlu_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.arabic_mmlu_scenario.ArabicMMLUScenario")

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="The following are multiple choice questions. Answer the last question. Respond only with only a single letter corresponding to your choice.",  # noqa: E501
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="arabic_mmlu",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arabic_mmlu"],
    )


@run_spec_function("alghafa")
def get_alghafa_spec(subset: str) -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.alghafa_scenario.AlGhafaScenario", args={"subset": subset}
    )

    adapter_spec = get_multiple_choice_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT,
        instructions="The following are multiple choice questions. Answer the last question. Respond only with only a single letter corresponding to your choice.",  # noqa: E501
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name=f"alghafa:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["alghafa", f"alghafa_{subset}"],
    )


@run_spec_function("aratrust")
def get_aratrust_spec() -> RunSpec:
    """EXPERIMENTAL: This run spec here may have future reverse incompatible changes."""
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.aratrust_scenario.AraTrustScenario")

    adapter_spec = get_generation_adapter_spec(
        instructions="The following are multiple choice questions. Answer the last question. Respond only with only a single letter corresponding to your choice.",  # noqa: E501
        input_noun="Question",
        output_noun="Answer",
    )

    return RunSpec(
        name="aratrust",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["aratrust"],
    )
