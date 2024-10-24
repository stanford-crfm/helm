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
    """Simple multiple-choice question answering scenario for tutorials and debugging.

    The task is to answer questions about whether two-digit numbers are even or odd.

    Example:

        Answer the following questions with a single letter only.

        Question: Is 24 even or odd?
        A. Even
        B. Odd
        Answer: A"""

    NUM_SPEAKERS = 60
    NUM_TRIALS = 50
    WAV_URL_TEMPLATE = r"https://github.com/soerenab/AudioMNIST/raw/544b0f4bc65227e54332e665d5e02c24be6732c2/data/{speaker_id}/{digit}_{speaker_id}_{trial_index}.wav"  # noqa: E501

    name = "audio_mnist"
    description = "Classify an audio sample of a spoken digit"
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
