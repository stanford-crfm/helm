"""Scenarios for audio models"""

from typing import List
from scipy.io.wavfile import write
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
from helm.common.general import ensure_directory_exists


class AudioMNISTScenario(Scenario):
    """AudioMNIST

    The AudioMNIST (Becker et al, 2023) dataset consists of a dataset of 30000 audio samples of
    spoken digits (0-9) of 60 different speakers. The task is to classify the digit from the
    audio sample.

    Paper: https://arxiv.org/abs/1807.03418
    Code: https://github.com/soerenab/AudioMNIST

    Citation:
    @article{audiomnist2023,
        title = {AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark},
        journal = {Journal of the Franklin Institute},
        year = {2023},
        issn = {0016-0032},
        doi = {https://doi.org/10.1016/j.jfranklin.2023.11.038},
        url = {https://www.sciencedirect.com/science/article/pii/S0016003223007536},
        author = {Sören Becker and Johanna Vielhaben and Marcel Ackermann and Klaus-Robert Müller and Sebastian Lapuschkin and Wojciech Samek},
        keywords = {Deep learning, Neural networks, Interpretability, Explainable artificial intelligence, Audio classification, Speech recognition},
    }
    """  # noqa: E501

    name = "audio_mnist"
    description = "Classify an audio sample of a spoken digit ([Becker et al, 2023](https://arxiv.org/abs/1807.03418))."
    tags = ["audio", "classification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        wav_save_dir: str = os.path.join(output_path, "wav_files")
        ensure_directory_exists(wav_save_dir)
        for row in tqdm(load_dataset("flexthink/audiomnist", cache_dir=output_path, split=TEST_SPLIT)):
            local_audio_path = os.path.join(wav_save_dir, row["audio"]["path"])
            audio_array = row["audio"]["array"]
            if not os.path.exists(local_audio_path):
                write(local_audio_path, row["audio"]["sampling_rate"], audio_array)
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/mpeg", location=local_audio_path)])
            )
            references = [Reference(Output(text=str(row["digit"])), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
