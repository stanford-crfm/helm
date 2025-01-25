from helm.benchmark.adaptation.common_adapter_specs import get_generation_adapter_spec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("bird_sql_dev")
def get_bird_sql_dev() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.bird_sql_dev_scenario.BIRDSQLDevScenario")

    adapter_spec = get_generation_adapter_spec(
        input_noun=None,
        output_noun=None,
        max_tokens=1024,
        stop_sequences=[],
    )

    return RunSpec(
        name="bird_sql_dev",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_exact_match_metric_specs(),
        groups=["bird_sql_dev"],
    )
