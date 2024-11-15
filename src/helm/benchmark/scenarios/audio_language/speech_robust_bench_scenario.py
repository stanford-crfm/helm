"""Scenarios for audio models"""

from typing import List
import os

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


class SpeechRobustBenchScenario(Scenario):
    """Speech Robust Bench Scenario

    Speech Robust Bench (Shah et al, 2024) is a comprehensive benchmark for evaluating
    the robustness of ASR models to diverse corruptions. SRB is composed of 114 input
    perturbations which simulate an heterogeneous range of corruptions that ASR models
    may encounter when deployed in the wild. In this scenario, we select four subsets
    in the benchmark for evaluation.

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

    # Select four subsets of the dataset for the benchmark
    SUBJECTS_DICT = {
        "accented_cv": {"name": "accented_cv", "split": TEST_SPLIT, "type": "audio/mpeg"},
        "accented_cv_es": {"name": "accented_cv_es", "split": TEST_SPLIT, "type": "audio/mpeg"},
        "chime_far": {"name": "chime", "split": "farfield", "type": "audio/wav"},
        "chime_near": {"name": "chime", "split": "nearfield", "type": "audio/wav"},
        "ami_far": {"name": "in-the-wild-AMI", "split": "farfield", "type": "audio/wav"},
        "ami_near": {"name": "in-the-wild-AMI", "split": "nearfield", "type": "audio/wav"},
    }

    name = "speech_robust_bench"
    description = (
        "Speech recognition for 4 datasets with a wide range of corruptions"
        "([Shah et al, 2024](https://arxiv.org/abs/2403.07937))."
    )
    tags: List[str] = ["audio", "recognition", "robustness", "multilinguality"]

    def __init__(self, subject: str) -> None:
        super().__init__()

        self._subject = subject
        if self._subject not in SpeechRobustBenchScenario.SUBJECTS_DICT.keys():
            raise ValueError(f"Invalid subject. Valid subjects are: {SpeechRobustBenchScenario.SUBJECTS_DICT.keys()}")

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        subject_name = SpeechRobustBenchScenario.SUBJECTS_DICT[self._subject]["name"]
        subject_split = SpeechRobustBenchScenario.SUBJECTS_DICT[self._subject]["split"]
        subject_type = SpeechRobustBenchScenario.SUBJECTS_DICT[self._subject]["type"]
        audio_save_dir = os.path.join(output_path, "audio_files")
        for row in tqdm(
            load_dataset(
                SpeechRobustBenchScenario.HF_DATASET_NAME,
                name=subject_name,
                cache_dir=output_path,
                split=subject_split,
            )
        ):
            local_audio_path = os.path.join(audio_save_dir, row["audio"]["path"])
            ensure_audio_file_exists_from_array(local_audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])
            answer = row["text"]
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type=subject_type, location=local_audio_path)])
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
