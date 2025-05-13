from typing import List, Optional
from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL, AdapterSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_multiple_choice_classification_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def audio_classification_metric_specs() -> List[MetricSpec]:
    return get_multiple_choice_classification_metric_specs() + get_basic_metric_specs(
        ["exact_match", "quasi_exact_match"]
    )


def _get_multiple_choice_joint_adapter_spec(
    input_noun: Optional[str],
    output_noun: str,
    max_train_instances: int = 0,
    num_outputs: int = 1,
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
        global_prefix="",
        instructions="Answer the multiple choice question by just giving the letter of the correct answer "
        "and nothing else.",
        input_prefix=f"{input_noun}: " if input_noun is not None else "",
        input_suffix="\n",
        output_prefix=f"{output_noun}: ",
        output_suffix="\n",
        instance_prefix="\n",
        max_train_instances=max_train_instances,
        num_outputs=num_outputs,
        max_tokens=1,
        stop_sequences=["\n"],
        temperature=0.0,
        random=None,
    )


@run_spec_function("ultra_suite_classification")
def get_ultra_suite_classification_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_classification_scenario.UltraSuiteClassificationScenario",
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = audio_classification_metric_specs()
    run_spec_name: str = "ultra_suite_classification"
    return RunSpec(
        name=f"{run_spec_name}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )
