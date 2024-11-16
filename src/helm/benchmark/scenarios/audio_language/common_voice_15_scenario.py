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
from collections import OrderedDict
from tqdm import tqdm
from datasets import load_dataset
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.hierarchical_logger import hlog


class CommonVoice15Scenario(Scenario):
    """CommonVoice15 Scenario

    The most recent release of CommonVoice15 (Ardila et al, 2019) includes 114 languages. Over 50,000
    individuals have participated so far, resulting in 2,500 hours of collected audio. This is the largest
    audio corpus in the public domain for speech recognition, both in terms of number of hours and number
    of languages. The task is to recognize the speech from the audio sample.



    Paper: https://arxiv.org/abs/1912.06670
    Code: https://github.com/common-voice/common-voice

    Citation:
    @article{ardila2019common,
        title={Common voice: A massively-multilingual speech corpus},
        author={Ardila, Rosana and Branson, Megan and Davis, Kelly and
        Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais,
        Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
        journal={arXiv preprint arXiv:1912.06670},
        year={2019}
        }

    """

    HF_DATASET_NAME = "mozilla-foundation/common_voice_15_0"

    # Randomly selected 4 languages from 114 languages in the Common Voice 15 dataset following
    # Qwen2-Audio (https://arxiv.org/abs/2407.10759). The full language is:
    # https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0/blob/main/languages.py
    _COMMON_VOICE_TEST_LANG_TO_ID = OrderedDict(
        [
            ("English", "en"),
            ("Chinese_hk", "zh-HK"),
            ("German", "de"),
            ("French", "fr"),
        ]
    )

    name = "common_voice_15"
    description = "Speech recognition for 4 languages from 114 different languages in Common Voice 15 \
        ([Ardila et al, 2019](https://arxiv.org/abs/1912.06670))."
    tags: List[str] = ["audio", "recognition", "multilinguality"]

    def __init__(self, language: str) -> None:
        super().__init__()

        language = language.capitalize()
        if language not in CommonVoice15Scenario._COMMON_VOICE_TEST_LANG_TO_ID.keys():
            raise ValueError(
                f"Invalid language. Valid languages are: {CommonVoice15Scenario._COMMON_VOICE_TEST_LANG_TO_ID.keys()}"
            )

        self._language: str = language
        hlog(
            "You need to sign in Huggingface to download the dataset. Please remember "
            "to sign in to download the dataset."
        )

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        language_category = CommonVoice15Scenario._COMMON_VOICE_TEST_LANG_TO_ID[self._language]
        for row in tqdm(
            load_dataset(
                CommonVoice15Scenario.HF_DATASET_NAME,
                name=language_category,
                cache_dir=output_path,
                split=TEST_SPLIT,
                trust_remote_code=True,
            )
        ):
            local_audio_path = row["path"]
            answer = row["sentence"]
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/mpeg", location=local_audio_path)])
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
