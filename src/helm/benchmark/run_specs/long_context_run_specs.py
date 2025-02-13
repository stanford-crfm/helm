from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.metrics.common_metric_specs import get_open_ended_generation_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ruler_hotpotqa")
def get_ruler_hotpotqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ruler_qa_scenarios.RulerQAScenario", args={}
    )

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
        max_tokens=512,  # ?
        stop_sequences=[],
    )

    return RunSpec(
        name="ruler_hotpotqa",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["ruler_hotpotqa"],
    )

@run_spec_function("ruler_squad")
def get_ruler_squad_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.ruler_qa_scenarios.RulerQAScenario", args={"dataset": "squad"}
    )

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
        max_tokens=512,  # ?
        stop_sequences=[],
    )

    return RunSpec(
        name="ruler_squad",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs(),
        groups=["ruler_squad"],
    )

