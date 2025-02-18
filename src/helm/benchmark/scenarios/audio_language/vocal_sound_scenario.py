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

from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded
from helm.common.audio_utils import is_invalid_audio_file


class VocalSoundScenario(Scenario):
    """Vocal Sound Scenario

    The VocalSound (Gong et al, 2022) dataset consists of 21,000 crowdsourced recordings
    of laughter, sighs, coughs, throat clearing, sneezes, and sniffs from 3,365 unique subjects.
    The task is to classify the human behaviour from the audio sample.

    Paper: https://arxiv.org/abs/2205.03433
    Code: https://github.com/YuanGongND/vocalsound

    Citation:
    @INPROCEEDINGS{gong_vocalsound,
                author={Gong, Yuan and Yu, Jin and Glass, James},
                booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
                title={Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
                year={2022},
                pages={151-155},
                doi={10.1109/ICASSP43922.2022.9746828}
                }
    """  # noqa: E501

    DOWNLOADING_URL = "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip"

    name = "vocal_sound"
    description = "Classify an audio sample of a spoken digit ([Gong et al, 2022](https://arxiv.org/abs/2205.03433))."
    tags: List[str] = ["audio", "classification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        down_loading_path = os.path.join(output_path, "download")
        ensure_file_downloaded(VocalSoundScenario.DOWNLOADING_URL, down_loading_path, unpack=True)
        wav_save_dir = os.path.join(down_loading_path, "audio_16k")
        for file_name in tqdm(os.listdir(wav_save_dir)):
            local_audio_path: str = os.path.join(wav_save_dir, file_name)
            if not file_name.endswith(".wav") or is_invalid_audio_file(local_audio_path):
                continue

            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/wav", location=local_audio_path)])
            )

            answer: str = file_name.split("_")[-1].split(".")[0]
            if answer == "throatclearing":
                answer = "throat clearing"

            references = [Reference(Output(text=str(answer)), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
