from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL
from helm.benchmark.adaptation.common_adapter_specs import get_multiple_choice_separate_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("arc_challenge_pt")
def get_arc_challenge_pt_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.arc_challenge_pt_scenario.ARC_Challenge_PT_Scenario", args={}
    )

    adapter_spec = get_multiple_choice_separate_adapter_spec(
        method=ADAPT_MULTIPLE_CHOICE_SEPARATE_ORIGINAL,
    )

    return RunSpec(
        name="arc_challenge_pt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["arc_challenge_pt"],
    )
