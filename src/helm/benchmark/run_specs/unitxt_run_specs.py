import os

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.metrics.common_metric_specs import get_basic_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("unitxt")
def get_unitxt_spec(**kwargs) -> RunSpec:
    card = kwargs.get("card")
    recipe = kwargs.get("recipe")
    if not card and not recipe:
        raise Exception("Unitxt card or recipe must be specified")
    if os.environ.get("HELM_UNITXT_SHORTEN_RUN_SPEC_NAMES", "").lower() == "true":
        name_suffix = ",".join(
            [f"{key}={value}" for key, value in kwargs.items() if key not in ["template_card_index", "loader_limit"]]
        )
    else:
        name_suffix = ",".join([f"{key}={value}" for key, value in kwargs.items()])
    name = f"unitxt:{name_suffix}"
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.unitxt_scenario.UnitxtScenario", args=kwargs)
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        global_prefix="",
        global_suffix="",
        instructions="",
        input_prefix="",
        input_suffix="",
        reference_prefix="A. ",
        reference_suffix="",
        output_prefix="",
        output_suffix="",
        instance_prefix="",
        max_train_instances=0,
        num_outputs=1,
        temperature=0.0,
        max_tokens=512,
        stop_sequences=[],
    )
    return RunSpec(
        name=name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=[
            MetricSpec(class_name="helm.benchmark.metrics.unitxt_metrics.UnitxtMetric", args=kwargs),
        ]
        + get_basic_metric_specs([]),
        groups=[f"unitxt_{card or recipe}"],
    )
