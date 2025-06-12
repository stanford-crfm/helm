from typing import List, Optional
from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION_MULTIMODAL,
    ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
    AdapterSpec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
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


@run_spec_function("ultra_suite_classification")
def get_ultra_suite_classification_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_classification_scenario.UltraSuiteClassificationScenario",  # noqa: E501
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
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_disorder_breakdown_scenario.UltraSuiteDisorderBreakdownScenario",  # noqa: E501
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


# Makes the model transcribe the child's speech into text without assuming what the child is supposed to say
# if the transcription matches the prompt, then it is classified as typically developing
# otherwise, it is classified as having a speech disorder
@run_spec_function("ultra_suite_asr_classification")
def get_ultra_suite_asr_classification_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_asr_classification.UltraSuiteASRClassificationScenario",  # noqa: E501
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="""You are a highly experienced Speech-Language Pathologist (SLP). An audio recording is provided to you, typically consisting of a speech prompt from a pathologist followed by a child's repetition. Based on your expertise transcribe the child's speech into text. Do not make any assumptions about the words the child is expected to say. Only transcribe based on the words that the child actually says. Only respond with the text transcription, no other text or commentary.""",  # noqa: E501
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


# Makes the model transcribe the child's speech into text and is allowed to assume what the child is supposed to say
@run_spec_function("ultra_suite_asr_transcription")
def get_ultra_suite_asr_transcription_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_asr_classification.UltraSuiteASRClassificationScenario",  # noqa: E501
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="""You are a highly experienced Speech-Language Pathologist (SLP). An audio recording will be provided, typically consisting of a speech prompt from a pathologist followed by a child's repetition. Based on your expertise transcribe the child's speech into text. Try to understand what the child is expected to say. And only respond with the transcription of the child's speech. Not the pathologist's prompt or any other commentary. Only respond with the text transcription, no other text, commentary or punctuations.""",  # noqa: E501
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
        class_name="helm.benchmark.scenarios.audio_language.ultra_suite_disorder_symptoms_scenario.UltraSuiteDisorderSymptomsScenario",  # noqa: E501
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
