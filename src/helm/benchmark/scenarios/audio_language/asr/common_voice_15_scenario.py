"""Scenarios for audio models"""

from collections import OrderedDict
from datasets import load_dataset
from tqdm import tqdm
from typing import Any, Dict, List

from helm.benchmark.scenarios.scenario import (
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.benchmark.scenarios.audio_language.asr.asr_scenario import (
    ASRInstance,
    ASRScenario,
    SpeakerMetadata,
    Language,
    Country,
    Gender,
    Age,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.hierarchical_logger import hlog


class CommonVoice15Scenario(ASRScenario):
    """CommonVoice15 Scenario

    The most recent release of CommonVoice15 (Ardila et al, 2019) includes 114 languages. Over 50,000
    individuals have participated so far, resulting in 2,500 hours of collected audio. This is the largest
    audio corpus in the public domain for speech recognition, both in terms of number of hours and number
    of languages. The task is to recognize the speech from the audio sample.

    Paper: https://arxiv.org/abs/1912.06670
    Code: https://github.com/common-voice/common-voice
    Dataset: https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0

    Metadata:
    - gender: male, female, ''
    - age: teens, twenties, thirties, forties, fifties, sixties, seventies, eighties, nineties, ''
    - locale: us, uk, fr, es, de, it, nl, pt, ru, cn, ..., ''

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

        self._speaker_langauge = Language(language.lower())
        self._language: str = language
        hlog(
            "You need to sign in Huggingface to download the dataset. Please remember "
            "to sign in to download the dataset."
        )

    def get_gender(self, example: Dict[str, Any]) -> Gender:
        if not example["gender"]:
            return Gender.UNKNOWN

        gender: str = example["gender"]
        if gender == "other":
            return Gender.NON_BINARY
        return Gender(example["gender"])

    def get_age(self, example: Dict[str, Any]) -> Age:
        if not example["age"]:
            return Age.UNKNOWN

        age: str = example["age"]
        if age == "fourties":
            age = "forties"
        return Age(age)

    def get_country(self, example: Dict[str, Any]) -> Country:
        try:
            locale: str = example["locale"]
            return Country[locale.upper()]
        except KeyError:
            hlog(f"Unknown country. Using unknown: {example}")
            return Country.UNKNOWN

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
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/mpeg", location=row["path"])])
            )
            references = [Reference(Output(text=row["sentence"]), tags=[CORRECT_TAG])]

            instances.append(
                ASRInstance(
                    input=input,
                    references=references,
                    split=TEST_SPLIT,
                    speaker_metadata=self.get_speaker_metadata(row),
                    language=self._speaker_langauge,
                )
            )
        return instances
