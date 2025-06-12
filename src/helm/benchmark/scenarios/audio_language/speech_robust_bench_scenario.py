"""Scenarios for audio models"""

from typing import List
import os
import json

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from tqdm import tqdm
from datasets import load_dataset
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.audio_utils import ensure_audio_file_exists_from_array
from helm.common.general import ensure_file_downloaded


class SpeechRobustBenchScenario(Scenario):
    """Speech Robust Bench Scenario

    Speech Robust Bench (Shah et al, 2024) is a comprehensive benchmark for evaluating
    the robustness of ASR models to diverse corruptions. SRB is composed of 114 input
    perturbations which simulate an heterogeneous range of corruptions that ASR models
    may encounter when deployed in the wild. In this scenario, we select four subsets
    in the benchmark for evaluation, each corresponds to a clean version of audio task.

    Paper: https://arxiv.org/abs/2403.07937
    Code: https://github.com/ahmedshah1494/speech_robust_bench

    Citation:
    @article{shah2024speech,
        title={Speech robust bench: A robustness benchmark for speech recognition},
        author={Shah, Muhammad A and Noguero, David Solans and Heikkila, Mikko A and Raj,
        Bhiksha and Kourtellis, Nicolas},
        journal={arXiv preprint arXiv:2403.07937},
        year={2024}
        }
    """

    HF_DATASET_NAME = "mshah1/speech_robust_bench"
    HF_MAPPING_URL = (
        "https://huggingface.co/datasets/PahaII/SRB_instance_key_mapping/resolve/main/srb_instance_keys.json"
    )

    # Select four subsets of the dataset for the benchmark
    SUBJECTS_DICT = {
        "ami_far": {
            "name": "in-the-wild-AMI",
            "split": "farfield",
            "type": "audio/wav",
            "mapping_key": "srb_aim_field_key2audio",
        },
        "ami_near": {
            "name": "in-the-wild-AMI",
            "split": "nearfield",
            "type": "audio/wav",
            "mapping_key": "srb_aim_field_key2audio",
        },
        "librispeech_gnoise": {
            "name": "librispeech_asr-test.clean_pertEval_500_30",
            "split": "gnoise.1",
            "type": "audio/mp3",
            "mapping_key": "srb_librispeech_noises_key2audio",
        },
        "librispeech_env_noise": {
            "name": "librispeech_asr-test.clean_pertEval_500_30",
            "split": "env_noise_esc50.1",
            "type": "audio/mp3",
            "mapping_key": "srb_librispeech_noises_key2audio",
        },
    }
    # There are 30 different perturbation samples for each LibriSpeech ID
    PERTURBATION_LEVELS = list(range(1, 31))
    name = "speech_robust_bench"
    description = (
        "Speech recognition for 4 datasets with a wide range of corruptions"
        "([Shah et al, 2024](https://arxiv.org/abs/2403.07937))."
    )
    tags: List[str] = ["audio", "recognition", "robustness", "multilinguality"]

    def __init__(self, subject: str, level: int) -> None:
        super().__init__()

        self._subject = subject
        if self._subject not in SpeechRobustBenchScenario.SUBJECTS_DICT.keys():
            raise ValueError(f"Invalid subject. Valid subjects are: {SpeechRobustBenchScenario.SUBJECTS_DICT.keys()}")
        self._level = level
        if self._level not in SpeechRobustBenchScenario.PERTURBATION_LEVELS:
            raise ValueError(f"Invalid level. Valid levels are: {SpeechRobustBenchScenario.PERTURBATION_LEVELS}")

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        subject_name = SpeechRobustBenchScenario.SUBJECTS_DICT[self._subject]["name"]
        subject_split = SpeechRobustBenchScenario.SUBJECTS_DICT[self._subject]["split"]
        subject_type = SpeechRobustBenchScenario.SUBJECTS_DICT[self._subject]["type"]
        subject_audio_type = subject_type.split("/")[-1]
        subject_mapping = SpeechRobustBenchScenario.SUBJECTS_DICT[self._subject]["mapping_key"]
        audio_save_dir = os.path.join(output_path, "audio_files")
        mapping_local_path = os.path.join(output_path, "srb_instance_keys.json")
        ensure_file_downloaded(source_url=SpeechRobustBenchScenario.HF_MAPPING_URL, target_path=mapping_local_path)
        mapping_keys = json.load(open(mapping_local_path))[subject_mapping][subject_split]
        meta_data = load_dataset(
            SpeechRobustBenchScenario.HF_DATASET_NAME,
            name=subject_name,
            cache_dir=output_path,
            split=subject_split,
        )
        for line_num in tqdm(list(mapping_keys.keys())):
            row = meta_data[int(mapping_keys[line_num][self._level - 1])]
            local_audio_name = f"{self._subject}_{subject_split}_{line_num}.{subject_audio_type}"
            local_audio_path = os.path.join(audio_save_dir, local_audio_name)
            ensure_audio_file_exists_from_array(local_audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])
            answer = row["text"].lower()
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type=subject_type, location=local_audio_path)])
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
