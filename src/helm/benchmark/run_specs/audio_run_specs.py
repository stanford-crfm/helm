"""Run spec functions for audio scenarios."""

from typing import List, Optional
from helm.benchmark.adaptation.adapter_spec import (
    AdapterSpec,
)
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION_MULTIMODAL
from helm.benchmark.metrics.common_metric_specs import (
    get_classification_metric_specs,
    get_exact_match_metric_specs,
    get_basic_metric_specs,
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


def _get_audio_recognition_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["wa_score", "ma_score", "wip_score", "ca_score"])


def _get_open_ended_generation_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4", "cider"]
    )


def _get_chinese_audio_recognition_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["chinese_wa_score", "chinese_ma_score", "chinese_wip_score", "chinese_ca_score"])


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


@run_spec_function("iemocap_audio")
def get_iemocap_audio_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.iemocap_audio_scenario.IEMOCAPAudioScenario"
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions='Classify the emotion of the speaker(s) in the audio as "angry", "happy", "neutral", or "sad". Answer with only the emotion.',  # noqa: E501
        max_tokens=5,
    )
    metric_specs = get_exact_match_metric_specs() + get_classification_metric_specs()
    return RunSpec(
        name="iemocap_audio",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["iemocap_audio"],
    )


@run_spec_function("meld_audio")
def get_meld_audio_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.meld_audio_scenario.MELDAudioScenario"
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions='Classify the emotion of the speaker in the audio as "anger", "disgust", "fear", "joy", "neutral", "sadness", or "surprise". Answer with only the emotion.',  # noqa: E501
        max_tokens=5,
    )
    metric_specs = get_exact_match_metric_specs() + get_classification_metric_specs()
    return RunSpec(
        name="meld_audio",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["meld_audio"],
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


@run_spec_function("vocal_sound")
def get_vocal_sound_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.vocal_sound_scenario.VocalSoundScenario",
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Listen to the audio and classify the speaker behavior. Choose only from these options:"
        '"Cough", "Laughter", "Sigh", "Sneeze", "Sniff", or "Throat clearing". Respond with just the behavior.',
        max_tokens=5,
    )
    metric_specs = get_exact_match_metric_specs() + get_classification_metric_specs()
    return RunSpec(
        name="vocal_sound",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["vocal_sound"],
    )


@run_spec_function("multilingual_librispeech")
def get_multilingual_librispeech_run_spec(language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.multilingual_librispeech_scenario."
        "MultilingualLibriSpeechScenario",
        args={"language": language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Listen to the audio and generate an accurate transcript of the spoken content. "
        "Respond with only the transcript text.",
        max_tokens=100,
    )
    if "chinese" in language.lower():
        metric_specs = _get_chinese_audio_recognition_metric_specs()
    else:
        metric_specs = _get_audio_recognition_metric_specs()
    return RunSpec(
        name="multilingual_librispeech",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["multilingual_librispeech"],
    )


@run_spec_function("fleurs")
def get_fleurs_run_spec(language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.fleurs_scenario.FLEURSScenario",
        args={"language": language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Listen to the audio and identify the language spoken. Choose from these"
        'options only: "Finnish", "Bulgarian", "Hebrew", "Zulu", "Bengali", "Thai",'
        '"Mandarin Chinese". Respond with just the language name.',
        max_tokens=5,
    )
    metric_specs = get_exact_match_metric_specs() + get_classification_metric_specs()
    return RunSpec(
        name="fleurs",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["fleurs"],
    )


@run_spec_function("audiocaps")
def get_audiocaps_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.audiocaps_scenario.AudioCapsScenario"
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Generate a caption for the following audio. The caption should be short and does "
        "not need to be a complete sentence.",
        max_tokens=50,
    )
    metric_specs: List[MetricSpec] = _get_open_ended_generation_metric_specs()
    return RunSpec(
        name="audiocaps",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["audiocaps"],
    )


@run_spec_function("common_voice_15")
def get_common_voice_15_run_spec(language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.common_voice_15_scenario.CommonVoice15Scenario",
        args={"language": language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Listen to the audio and generate an accurate transcript of the spoken content. "
        "Respond with only the transcript text.",
        max_tokens=100,
    )
    # Chinese characters are not supported in the default metrics
    if "chinese" in language.lower():
        metric_specs = _get_chinese_audio_recognition_metric_specs()
    else:
        metric_specs = _get_audio_recognition_metric_specs()
    return RunSpec(
        name="common_voice_15",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["common_voice_15"],
    )


@run_spec_function("speech_robust_bench")
def get_speech_robust_bench_run_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.speech_robust_bench_scenario.SpeechRobustBenchScenario",
        args={"subject": subject},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Listen to the audio and generate an accurate transcript of the spoken content. "
        "Respond with only the transcript text.",
        max_tokens=100,
    )
    metric_specs = _get_audio_recognition_metric_specs()
    return RunSpec(
        name="speech_robust_bench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["speech_robust_bench"],
    )


@run_spec_function("audio_pairs")
def get_audio_pairs_run_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.audio_pairs_scenario.AudioPAIRSScenario",
        args={"subject": subject},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Listen to the audio and answer the question with provided options.",
        max_tokens=5,
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs() + get_classification_metric_specs()
    return RunSpec(
        name="audio_pairs",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["audio_pairs"],
    )
