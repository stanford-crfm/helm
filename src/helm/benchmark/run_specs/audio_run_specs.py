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
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def _get_multimodal_generation_adapter_spec(
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


@run_spec_function("audio_mnist")
def get_audio_mnist_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.audio_scenarios.AudioMNISTScenario")
    adapter_spec = _get_multimodal_generation_adapter_spec(
        instructions="Classify the spoken digit. Respond with only a single digit."
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
