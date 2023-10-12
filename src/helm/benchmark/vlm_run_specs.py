from typing import List, Optional

from .adaptation.adapter_spec import AdapterSpec
from .adaptation.adapters.adapter_factory import ADAPT_GENERATION_MULTIMODAL
from .metrics.metric import MetricSpec
from .run_specs import run_spec_function, get_exact_match_metric_specs
from .runner import RunSpec
from .scenarios.scenario import ScenarioSpec


############################################################
# Prototypical adapter specs for VLM evaluation


def get_vlm_generation_adapter_spec(
    instructions: str = "",
    input_prefix: str = "",
    input_suffix: str = "",
    output_prefix: str = "",
    output_suffix: str = "",
    max_train_instances: int = 0,
    max_tokens: int = 100,
    stop_sequences: Optional[List[str]] = None,
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        global_prefix="",
        instructions=instructions,
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        output_prefix=output_prefix,
        output_suffix=output_suffix,
        instance_prefix="\n",
        max_train_instances=max_train_instances,
        num_outputs=1,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences if stop_sequences is not None else [],
        random=None,
    )


############################################################
# VHELM run specs


@run_spec_function("vqa")
def get_vqa_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.vision_language.vqa_scenario.VQAScenario", args={}
    )

    # TODO: finalize the adapter spec parameters once we add more models
    adapter_spec: AdapterSpec = get_vlm_generation_adapter_spec(
        input_prefix="User: ",
        input_suffix="<end_of_utterance>",
        output_prefix="\nAssistant: ",
        output_suffix="<end_of_utterance>",
        max_train_instances=3,
        stop_sequences=["<end_of_utterance>"],
    )

    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()

    run_spec_name: str = "vqa"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )
