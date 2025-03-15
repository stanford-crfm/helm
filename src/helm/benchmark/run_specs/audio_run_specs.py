"""Run spec functions for audio scenarios."""

from typing import List, Optional
from helm.benchmark.adaptation.adapter_spec import (
    AdapterSpec,
)
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION_MULTIMODAL,
    ADAPT_MULTIPLE_CHOICE_JOINT_MULTIMODAL,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_classification_metric_specs,
    get_exact_match_metric_specs,
    get_generative_harms_metric_specs,
    get_basic_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


########################################################################################################################
# Constants
ASR_INSTRUCTIONS: str = (
    "Listen to the audio and transcribe the spoken content to text. "
    "Respond with only the transcript text and nothing else."
)


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


########################################################################################################################
# MetricSpecs


def get_machine_translation_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.machine_translation_metrics.MachineTranslationMetric")]


def _get_audio_recognition_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["wer_score", "mer_score", "wip_score", "cer_score"])


def _get_open_ended_generation_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(
        ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4", "cider"]
    )


def _get_chinese_audio_recognition_metric_specs() -> List[MetricSpec]:
    return get_basic_metric_specs(["chinese_wer_score", "chinese_mer_score", "chinese_wip_score", "chinese_cer_score"])


def _get_gpt4_critique_metric_specs(num_respondents: int, max_tokens: int) -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.gpt4_audio_critique_metrics.GPT4AudioCritiqueMetric",
            args={
                "num_respondents": num_respondents,
                "max_tokens": max_tokens,
            },
        )
    ]


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
    run_spec_name: str = "audio_mnist"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
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
    run_spec_name: str = "iemocap_audio"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
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
    run_spec_name: str = "meld_audio"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mutox")
def get_mutox_audio_run_spec(language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.mutox_scenario.MuToxScenario",
        args={"language": language},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "mutox"
    return RunSpec(
        name=f"{run_spec_name}:language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("mustard")
def get_mustard_audio_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.audio_language.mustard_scenario.MUStARDScenario")
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "mustard"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("voice_jailbreak_attacks")
def get_voice_jailbreak_attacks_run_spec(subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.voice_jailbreak_attacks_scenario."
        "VoiceJailbreakAttacksScenario",
        args={"subset": subset},
    )
    adapter_spec = _get_generation_adapter_spec(max_tokens=1024)
    metric_specs: List[MetricSpec] = get_generative_harms_metric_specs(
        include_basic_metrics=True, include_generative_harms_metrics=True
    )

    run_spec_name: str = "voice_jailbreak_attacks"
    return RunSpec(
        name=f"{run_spec_name}:subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("covost2")
def get_covost2_run_spec(source_language: str, target_language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.covost2_scenario.CoVoST2Scenario",
        args={"source_language": source_language, "target_language": target_language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=f"Translate from {source_language} to {target_language}. "
        "Just give the translation and nothing else.",
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
    run_spec_name: str = "vocal_sound"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("multilingual_librispeech")
def get_multilingual_librispeech_run_spec(language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.multilingual_librispeech_scenario."
        "MultilingualLibriSpeechScenario",
        args={"language": language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    if "chinese" in language.lower():
        metric_specs = _get_chinese_audio_recognition_metric_specs()
    else:
        metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "multilingual_librispeech"
    return RunSpec(
        name=f"{run_spec_name}:language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("fleurs")
def get_fleurs_run_spec(language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.fleurs_scenario.FLEURSScenario",
        args={"language": language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    # Chinese characters are not supported in the default metrics
    if "chinese" in language.lower():
        metric_specs = _get_chinese_audio_recognition_metric_specs()
    else:
        metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "fleurs"
    return RunSpec(
        name=f"{run_spec_name}:language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("fleurs_fairness")
def get_fleurs_fairness_run_spec(gender: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.fleurs_fairness_scenario.FLEURSFairnessScenario",
        args={"gender": gender},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "fleurs_fairness"
    return RunSpec(
        name=f"{run_spec_name}:gender={gender}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("audiocaps")
def get_audiocaps_run_spec(num_respondents: int = 1) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.audiocaps_scenario.AudioCapsScenario"
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="Generate a caption describing what you hear in the following audio. "
        "The caption should be short and does not need to be a complete sentence. Respond with "
        "only the caption and nothing else.",
        max_tokens=50,
    )
    metric_specs: List[MetricSpec] = (
        _get_gpt4_critique_metric_specs(
            num_respondents=num_respondents,
            max_tokens=200,
        )
        + _get_open_ended_generation_metric_specs()
    )
    run_spec_name: str = "audiocaps"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("voxceleb2")
def get_voxceleb2_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.voxceleb2_scenario.VoxCeleb2Scenario"
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "voxceleb2"
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("common_voice_15")
def get_common_voice_15_run_spec(language: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.common_voice_15_scenario.CommonVoice15Scenario",
        args={"language": language},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    # Chinese characters are not supported in the default metrics
    if "chinese" in language.lower():
        metric_specs = _get_chinese_audio_recognition_metric_specs()
    else:
        metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "common_voice_15"
    return RunSpec(
        name=f"{run_spec_name}:language={language}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("speech_robust_bench")
def get_speech_robust_bench_run_spec(subject: str, level: int) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.speech_robust_bench_scenario.SpeechRobustBenchScenario",
        args={"subject": subject, "level": level},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "speech_robust_bench"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject},level={level}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
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
    run_spec_name: str = "audio_pairs"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("casual_conversations2")
def get_casual_conversations2_run_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.casual_conversations2_scenario."
        "CasualConversations2Scenario",
        args={"subject": subject},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs: List[MetricSpec] = get_exact_match_metric_specs()
    run_spec_name: str = "casual_conversations2"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("air_bench_chat")
def get_air_bench_chat_run_spec(subject: str, num_respondents: int = 1) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.air_bench_chat_scenario." "AirBenchChatScenario",
        args={"subject": subject},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions="",
        max_tokens=200,
    )
    metric_specs: List[MetricSpec] = (
        _get_gpt4_critique_metric_specs(
            num_respondents=num_respondents,
            max_tokens=200,
        )
        + _get_open_ended_generation_metric_specs()
    )
    run_spec_name: str = "air_bench_chat"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("ami")
def get_ami_run_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.ami_scenario.AMIScenario",
        args={"subject": subject},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "ami"
    return RunSpec(
        name=f"{run_spec_name}:subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("librispeech")
def get_librispeech_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.librispeech_scenario.LibriSpeechScenario",
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "librispeech"
    return RunSpec(
        name=f"{run_spec_name}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("librispeech_fairness")
def get_librispeech_fairness_run_spec(gender: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.librispeech_fairness_scenario.LibriSpeechFairnessScenario",
        args={"gender": gender},
    )
    adapter_spec = _get_generation_adapter_spec(
        instructions=ASR_INSTRUCTIONS,
        max_tokens=100,
    )
    metric_specs = _get_audio_recognition_metric_specs()
    run_spec_name: str = "librispeech_fairness"
    return RunSpec(
        name=f"{run_spec_name}:gender={gender}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("air_bench_foundation")
def get_air_bench_foundation_run_spec(subject: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.air_bench_foundation_scenario.AirBenchFoundationScenario",
        args={"subject": subject},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs = get_exact_match_metric_specs()
    run_spec_name: str = "air_bench_foundation"
    return RunSpec(
        name=f"{run_spec_name},subject={subject}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )


@run_spec_function("parade")
def get_parade_run_spec(voice: str, subset: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.audio_language.parade_scenario.PARADEScenario",
        args={"subset": subset, "voice": voice},
    )
    adapter_spec: AdapterSpec = _get_multiple_choice_joint_adapter_spec(
        input_noun=None, output_noun="Answer", max_train_instances=0
    )
    metric_specs = get_exact_match_metric_specs()
    run_spec_name: str = "parade"
    return RunSpec(
        name=f"{run_spec_name},voice={voice},subset={subset}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=[run_spec_name],
    )
