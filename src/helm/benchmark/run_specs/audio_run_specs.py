"""Run spec functions for audio scenarios."""

from typing import List, Optional
from helm.benchmark.adaptation.adapter_spec import (
    AdapterSpec,
)
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION_MULTIMODAL
from helm.benchmark.metrics.common_metric_specs import (
    get_classification_metric_specs,
    get_exact_match_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


########################################################################################################################
#  AdapterSpecs


def _get_generation_adapter_spec(
    max_tokens: int,
    instructions: str = "",
    max_train_instances: int = 0,
    temperature: float = 0.0,
    stop_sequences: Optional[List[str]] = None,
) -> AdapterSpec:
    return AdapterSpec(
        method=ADAPT_GENERATION_MULTIMODAL,
        instructions=instructions,
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        instance_prefix="",
        max_train_instances=max_train_instances,
        num_outputs=1,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences if stop_sequences is not None else [],
    )


########################################################################################################################
# MetricSpecs


def get_machine_translation_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.machine_translation_metrics.MachineTranslationMetric")]


########################################################################################################################
# RunSpecs


@run_spec_function("audio_mnist")
def get_audio_mnist_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.audio_mnist_scenario.AudioMNISTScenario"
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Classify the spoken digit. Respond with only a single digit.",
        max_tokens=5,
    )
    metric_specs = get_exact_match_metric_specs() + get_classification_metric_specs()
    return RunSpec(
        name="audio_mnist",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["audio_mnist"],
    )


@run_spec_function("covost2")
def get_covost2_run_spec(source_language: str, target_language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.covost2_scenario.CoVoST2Scenario",
        args={"source_language": source_language, "target_language": target_language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=f"Translate from {source_language} to {target_language}.",
        max_tokens=50,
    )
    metric_specs = get_machine_translation_metric_specs()
    return RunSpec(
        name=f"covost2:source_language={source_language},target_language={target_language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["covost2"],
    )
