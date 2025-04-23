from typing import List
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.run_specs.audio_run_specs import _get_multiple_choice_joint_adapter_spec
from helm.benchmark.scenarios.scenario import ScenarioSpec


@run_spec_function("ultra_suite_classification")
def get_ultra_suite_classification_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.speech_disorder_scenario.UltraSuiteClassificationScenario",
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "ultra_suite_classification"
    return RunSpec(
        name=f"{run_spec_name}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )
