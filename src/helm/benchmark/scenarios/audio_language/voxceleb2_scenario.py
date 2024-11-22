from typing import List
import os
import os.path as osp

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
import pandas as pd
from glob import glob
from pydub import AudioSegment
from multiprocessing import Pool


def _m4a_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio.export(output_path, format="wav")


def _preprocess_single_sample(audio_path):
    assert osp.exists(audio_path), f"Audio file does not exist at path: {audio_path}"
    audio_path_wav = audio_path[:-3] + "wav"
    audio = AudioSegment.from_file(audio_path, format="m4a")
    audio.export(audio_path_wav, format="wav")


class VoxCeleb2Scenario(Scenario):
    """VoxCeleb2

    VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted from
    interview videos uploaded to YouTube.

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

    name = "voxceleb2"
    description = "A large-scale dataset of about 46K audio clips to human-written text pairs \
        ([Kim et al, 2019](https://aclanthology.org/N19-1011.pdf))."
    tags: List[str] = ["audio", "identification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        data_root = osp.join(output_path, "data/test/aac")
        ensure_file_downloaded(source_url=VoxCeleb2Scenario.DOWNLOADING_URL, target_path=data_root, unpack=True)
        df = pd.read_csv(VoxCeleb2Scenario.REFERENCE_URL, sep=",")

        df["first"] = df["first"].apply(lambda x: osp.join(data_root, x))
        df["second"] = df["second"].apply(lambda x: osp.join(data_root, x))

        all_paths = set(df["first"].to_list() + df["second"].to_list())
        with Pool(processes=4) as pool:
            list(tqdm(pool.imap(_preprocess_single_sample, all_paths), total=len(all_paths)))

        instances = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            first = row["first"][:-3] + "wav"
            second = row["second"][:-3] + "wav"
            same = "True" if row["same"] else "False"

            input = Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(content_type="audio/wav", location=first),
                        MediaObject(content_type="audio/wav", location=second),
                    ]
                )
            )

            references = [Reference(Output(text=same), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))

        return instances
