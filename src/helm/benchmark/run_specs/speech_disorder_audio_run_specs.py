from typing import List
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.common_metric_specs import audio_classification_metric_specs, get_basic_generation_metric_specs, get_exact_match_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.run_specs.audio_run_specs import _get_audio_recognition_metric_specs, _get_generation_adapter_spec, _get_multiple_choice_joint_adapter_spec, _get_open_ended_generation_metric_specs
from helm.benchmark.scenarios.scenario import ScenarioSpec


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

@run_spec_function("ultra_suite_classification_breakdown")
def get_ultra_suite_disorder_breakdown_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_disorder_breakdown_scenario.UltraSuiteDisorderBreakdownScenario",
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = audio_classification_metric_specs()
    run_spec_name: str = "ultra_suite_classification_breakdown"
    return RunSpec(
        name=f"{run_spec_name}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )

@run_spec_function("ultra_suite_asr_classification")
def get_ultra_suite_asr_classification_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_asr_classification.UltraSuiteASRClassificationScenario",
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="""You are a highly experienced Speech-Language Pathologist (SLP). 
            An audio recording is provided to you, typically consisting of a speech prompt 
            from a pathologist followed by a child's repetition. 
            Based on your expertise transcribe the child's speech into text.
            Do not make any assumptions about the words the child is expected to say.
            Only transcribe based on the words that the child actually says.
            Only respond with the text transcription, no other text or commentary.
            """,
        max_tokens=10,
    )   
    metric_specs: List[MetricSpec] = audio_classification_metric_specs()
    run_spec_name: str = "ultra_suite_asr_classification"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )

@run_spec_function("ultra_suite_asr_transcription")
def get_ultra_suite_asr_transcription_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_asr_classification.UltraSuiteASRClassificationScenario",
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="""You are a highly experienced Speech-Language Pathologist (SLP). 
            An audio recording will be provided, typically consisting of a speech prompt 
            from a pathologist followed by a child's repetition. 
            Based on your expertise transcribe the child's speech into text.
            Do not make any assumptions about the words the child is expected to say.
            Only transcribe based on the words that the child actually says.
            And only respond with the transcription of the child's speech. Not the pathologist's prompt or any other commentary.
            Only respond with the text transcription, no other text, commentary or punctuations.
            """,
        max_tokens=50,
    )   
    metric_specs: List[MetricSpec] = get_basic_generation_metric_specs(["wer_score", "mer_score", "wip_score"])
    run_spec_name: str = "ultra_suite_asr_transcription"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )

@run_spec_function("ultra_suite_disorder_symptoms")
def get_ultra_suite_disorder_symptoms_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_disorder_tags_scenario.UltraSuiteDisorderTagsScenario",
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = audio_classification_metric_specs()
    run_spec_name: str = "ultra_suite_disorder_symptoms"
    return RunSpec(
        name=f"{run_spec_name}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )