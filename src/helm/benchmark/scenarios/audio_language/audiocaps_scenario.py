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


class AudioCapsScenario(Scenario):
    """AudioCaps

    AudioCaps is a large-scale dataset of about 46K audio clips to human-written text pairs collected
    via crowdsourcing on the AudioSet dataset, which covers a wide range of human and animal sounds,
    musical instruments and genres, and common everyday environmental sounds.

    Paper: https://aclanthology.org/N19-1011.pdf
    Code: https://github.com/cdjkim/audiocaps

    Citation:
    @inproceedings{audiocaps,
        title={AudioCaps: Generating Captions for Audios in The Wild},
        author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
        booktitle={NAACL-HLT},
        year={2019}
        }
    """

    DOWNLOADING_URL = "https://huggingface.co/datasets/Olivia714/audiocaps/resolve/main/wav_files.zip"
    REFERENCE_URL = "https://huggingface.co/datasets/Olivia714/audiocaps/resolve/main/test.csv"

    name = "audiocaps"
    description = "A large-scale dataset of about 46K audio clips to human-written text pairs \
        ([Kim et al, 2019](https://aclanthology.org/N19-1011.pdf))."
    tags: List[str] = ["audio", "captioning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        data_dir: str = os.path.join(output_path, "wav_files")
        ensure_file_downloaded(source_url=AudioCapsScenario.DOWNLOADING_URL, target_path=data_dir, unpack=True)
        for _, row in tqdm(pd.read_csv(AudioCapsScenario.REFERENCE_URL, sep=",").iterrows()):
            audiocap_id = row["audiocap_id"]
            audio_path: str = os.path.join(data_dir, f"{audiocap_id}.wav")
            assert os.path.exists(audio_path), f"Audio file does not exist at path: {audio_path}"
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/wav", location=audio_path)])
            )
            references = [Reference(Output(text=str(row["caption"])), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
