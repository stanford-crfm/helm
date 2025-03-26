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
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.audio_utils import use_ffmpeg_to_convert_audio_file
import pandas as pd


class VoxCeleb2Scenario(Scenario):
    """VoxCeleb2

    VoxCeleb2 is an audio-visual dataset consisting of short clips of human speech, extracted from
    interview videos uploaded to YouTube. This dataset contains over a million utterances from over
    6,000 speakers.

    Paper: https://www.robots.ox.ac.uk/~vgg/publications/2018/Chung18a/chung18a.pdf

    Citation:
    @inproceedings{Chung18b,
        author = "Chung, J.~S. and Nagrani, A. and Zisserman, A.",
        title = "VoxCeleb2: Deep Speaker Recognition",
        booktitle = "INTERSPEECH",
        year = "2018",
    }
    """

    DOWNLOADING_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_test_aac.zip"
    REFERENCE_URL = (
        "https://huggingface.co/datasets/LAOS-Y/VoxCeleb2-AudioIdentity/resolve/main/voxceleb2_audioidentity.csv"
    )
    IDENTITY_INSTRUCTION = (
        "Listen to the audio and take your best guess to determine if the two speakers are the same person. "
        "Give just the letter of your answer and nothing else."
    )

    name = "voxceleb2"
    description = (
        "A large-scale dataset of over a million utterances from over 6,000 speakers with their"
        "gender, race, identity information"
        "([Chung et al, 2018](https://www.robots.ox.ac.uk/~vgg/publications/2018/Chung18a/chung18a.pdf))."
    )
    tags: List[str] = ["audio", "identification"]
    options: List[str] = ["Yes", "No"]

    def _convert_answer_to_label(self, answer: bool) -> str:
        if answer:
            return "A"
        else:
            return "B"

    def _reformat_and_convert_audio_file(
        self, ori_file_path: str, tgt_audio_data_path: str, audio_data_path: str
    ) -> str:
        tgt_audio_path = os.path.join(tgt_audio_data_path, ori_file_path.split(".m4a")[0] + ".wav")
        ensure_directory_exists(os.path.dirname(tgt_audio_path))
        use_ffmpeg_to_convert_audio_file(os.path.join(audio_data_path, ori_file_path), tgt_audio_path)
        return tgt_audio_path

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        audio_data_path = os.path.join(output_path, "audio_files")
        tgt_audio_data_path = os.path.join(output_path, "tgt_audio_files")
        ensure_file_downloaded(source_url=VoxCeleb2Scenario.DOWNLOADING_URL, target_path=audio_data_path, unpack=True)
        annotations = pd.read_csv(VoxCeleb2Scenario.REFERENCE_URL, sep=",")
        instances = []
        for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
            tgt_first_audio_path = self._reformat_and_convert_audio_file(
                row["first"], tgt_audio_data_path, audio_data_path
            )
            tgt_second_audio_path = self._reformat_and_convert_audio_file(
                row["second"], tgt_audio_data_path, audio_data_path
            )

            answer = self._convert_answer_to_label(row["same"])
            # The given correct answer is a letter, but we need an index
            correct_answer_index: int = ord(answer) - ord("A")
            references: List[Reference] = []
            for i, option in enumerate(self.options):
                reference: Reference
                is_correct: bool = i == correct_answer_index
                reference = Reference(Output(text=option), tags=[CORRECT_TAG] if is_correct else [])
                references.append(reference)

            input = Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(content_type="audio/wav", location=tgt_first_audio_path),
                        MediaObject(content_type="audio/wav", location=tgt_second_audio_path),
                        MediaObject(content_type="text/plain", text=self.IDENTITY_INSTRUCTION),
                    ]
                )
            )
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))

        return instances
