"""Run spec functions for Vietnam WVS cultural alignment evaluation."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("vietnam_wvs")
def get_vietnam_wvs_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.viet_wvs_scenario.VietnamWVSScenario",
        args={
            "num_personas": 300,
            "num_question_variants": 4,
            "include_examples": True,
        }
    )
    
    # Custom adapter spec that uses numbers directly (not letters)
    adapter_spec = AdapterSpec(
        method="generation",  # Use generation, not multiple_choice
        global_prefix="",
        global_suffix="",
        instructions="Please respond as the Vietnamese persona described.\n",
        input_prefix="Question: ",
        input_suffix="\n",
        output_prefix="Answer: ",
        output_suffix="\n",
        instance_prefix="\n",
        # max_train_instances=3,
        # max_eval_instances=100,
        temperature=0.0,
        max_tokens=3,  # Just enough for a numeric response
        stop_sequences=["\n"],
    )
    
    # Use exact match metrics
    metric_specs = get_exact_match_metric_specs()
    
    return RunSpec(
        name="vietnam_wvs",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["cultural_alignment", "vietnam_wvs"],
    )