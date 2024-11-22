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
import pandas as pd
from glob import glob
from pydub import AudioSegment
from multiprocessing import Pool


def _m4a_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio.export(output_path, format="wav")


def _process_single_sample(audio_path_gender_pair):
    audio_path, gender = audio_path_gender_pair
    audio_path_wav = audio_path[:-3] + "wav"
    _m4a_to_wav(audio_path, audio_path_wav)
    input = Input(
        multimedia_content=MultimediaObject([MediaObject(content_type="audio/wav", location=audio_path_wav)])
    )
    references = [Reference(Output(text=gender), tags=[CORRECT_TAG])]
    return Instance(input=input, references=references, split=TEST_SPLIT)


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
    REFERENCE_URL = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_meta.csv"

    name = "voxceleb2"
    description = "A large-scale dataset of about 46K audio clips to human-written text pairs \
        ([Kim et al, 2019](https://aclanthology.org/N19-1011.pdf))."
    tags: List[str] = ["audio", "classification"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        data_root = os.path.join(output_path, "data/test/aac")
        ensure_file_downloaded(source_url=VoxCeleb2Scenario.DOWNLOADING_URL, target_path=data_root, unpack=True)
        df = pd.read_csv(VoxCeleb2Scenario.REFERENCE_URL, sep=" ,")
        df = df[df["Set"] == "test"]
        df = df[df["VoxCeleb2 ID"].apply(lambda x: x not in ["id04170", "id05348"])]

        audio_path_gender_pairs = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            vox_celeb2_id = row["VoxCeleb2 ID"]
            gender = "Male" if row["Gender"] == "m" else "Female"
            audio_dir = os.path.join(data_root, vox_celeb2_id)
            assert os.path.exists(audio_dir), f"Audio file does not exist at path: {audio_dir}"

            audio_paths = glob(os.path.join(audio_dir, "**/*.m4a"), recursive=True)
            audio_paths = sorted(audio_paths)

            audio_path_gender_pairs += [(audio_path, gender) for audio_path in audio_paths]

        with Pool(processes=4) as pool:
            instances = pool.map(_process_single_sample, audio_path_gender_pairs)

        return instances
