"""Scenarios for audio models"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.media_object import MediaObject, MultimediaObject


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

    NUM_SPEAKERS = 60
    NUM_TRIALS = 50
    WAV_URL_TEMPLATE = r"https://github.com/soerenab/AudioMNIST/raw/544b0f4bc65227e54332e665d5e02c24be6732c2/data/{speaker_id}/{digit}_{speaker_id}_{trial_index}.wav"  # noqa: E501

    name = "audio_mnist"
    description = "Classify an audio sample of a spoken digit ([Becker et al, 2023](https://arxiv.org/abs/1807.03418))."
    tags = ["audio", "classification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for digit in range(10):
            for speaker_index in range(AudioMNISTScenario.NUM_SPEAKERS):
                speaker_id = str(speaker_index).zfill(2)
                for trial_index in range(AudioMNISTScenario.NUM_TRIALS):
                    wav_url = AudioMNISTScenario.WAV_URL_TEMPLATE.format(
                        digit=digit, speaker_id=speaker_id, trial_index=trial_index
                    )
                    input = Input(
                        multimedia_content=MultimediaObject([MediaObject(content_type="audio/wav", location=wav_url)])
                    )
                    references = [Reference(Output(text=str(digit)), tags=[CORRECT_TAG])]
                    # Don't need train split because we're using zero-shot
                    instance = Instance(input=input, references=references, split=TEST_SPLIT)
                    instances.append(instance)
        return instances
