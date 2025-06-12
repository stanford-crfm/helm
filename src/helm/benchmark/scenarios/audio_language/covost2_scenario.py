from typing import Dict, List
import os

from datasets import load_dataset
from tqdm import tqdm

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


class CoVoST2Scenario(Scenario):
    """
    CoVost-2 is a large-scale multilingual speech translation corpus covering translations from 21 languages
    into English and from English into 15 languages.

    The dataset contains the audio, transcriptions, and translations in the following languages:
    French, German, Dutch, Russian, Spanish, Italian, Turkish, Persian, Swedish, Mongolian, Chinese,
    Welsh, Catalan, Slovenian, Estonian, Indonesian, Arabic, Tamil, Portuguese, Latvian, and Japanese.

    Paper: https://arxiv.org/abs/2007.10310
    Dataset: https://huggingface.co/datasets/facebook/covost2

    Requires downloading Common Voice Corpus 4 from https://commonvoice.mozilla.org/en/datasets

    Citation:
        @misc{wang2020covost2massivelymultilingual,
            title={CoVoST 2 and Massively Multilingual Speech-to-Text Translation},
            author={Changhan Wang and Anne Wu and Juan Pino},
            year={2020},
            eprint={2007.10310},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2007.10310},
        }
    """

    LANGUAGE_TO_CODE: Dict[str, str] = {
        "English": "en",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Chinese": "zh-CN",
        "Japanese": "ja",
        "Turkish": "tr",
        "Persian": "fa",
        "Arabic": "ar",
        "Dutch": "nl",
        "Swedish": "sv-SE",
        "Indonesian": "id",
        "Tamil": "ta",
        "Latvian": "lv",
        "Slovenian": "sl",
        "Welsh": "cy",
        "Mongolian": "mn",
        "Estonian": "et",
    }

    VALID_SUBSETS: List[str] = [
        "en_de",
        "en_tr",
        "en_fa",
        "en_sv-SE",
        "en_mn",
        "en_zh-CN",
        "en_cy",
        "en_ca",
        "en_sl",
        "en_et",
        "en_id",
        "en_ar",
        "en_ta",
        "en_lv",
        "en_ja",
        "fr_en",
        "de_en",
        "es_en",
        "ca_en",
        "it_en",
        "ru_en",
        "zh-CN_en",
        "pt_en",
        "fa_en",
        "et_en",
        "mn_en",
        "nl_en",
        "tr_en",
        "ar_en",
        "sv-SE_en",
        "lv_en",
        "sl_en",
        "ta_en",
        "ja_en",
        "id_en",
        "cy_en",
    ]

    name = "covost2"
    description = (
        "A large scale multilingual speech translation corpus ([Wang et al., 2017](https://arxiv.org/abs/2007.10310))."
    )
    tags = ["audio", "translation", "multilinguality"]

    def __init__(self, source_language: str, target_language: str) -> None:
        super().__init__()

        if (
            source_language not in CoVoST2Scenario.LANGUAGE_TO_CODE
            or target_language not in CoVoST2Scenario.LANGUAGE_TO_CODE
        ):
            raise ValueError(f"Invalid language. Valid languages are: {list(CoVoST2Scenario.LANGUAGE_TO_CODE.keys())}")

        # Get the corresponding language codes
        source_language_code: str = self.LANGUAGE_TO_CODE[source_language]
        target_language_code: str = self.LANGUAGE_TO_CODE[target_language]

        subset: str = f"{source_language_code}_{target_language_code}"
        if subset not in CoVoST2Scenario.VALID_SUBSETS:
            raise ValueError(f"Invalid subset: {subset}. Valid subsets are: {CoVoST2Scenario.VALID_SUBSETS}")

        self._subset: str = subset
        self._source_language: str = source_language

    def get_instances(self, output_path: str) -> List[Instance]:
        data_dir: str = os.path.join(output_path, self._source_language)
        assert os.path.exists(data_dir), (
            f"Download the {self._source_language} subset from Common Voice Corpus 4 "
            f"(https://commonvoice.mozilla.org/en/datasets) and unzip and place at {data_dir}."
        )

        instances: List[Instance] = []
        split: str = TEST_SPLIT
        for row in tqdm(
            load_dataset(
                "facebook/covost2",
                self._subset,
                cache_dir=output_path,
                data_dir=data_dir,
                split=split,
                trust_remote_code=True,
                revision="369b47c4c20aff1193b8edeeedc37d14ae28226b",
            )
        ):
            audio_path: str = row["file"]
            assert os.path.exists(audio_path), f"Audio file does not exist at path: {audio_path}"

            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/mpeg", location=audio_path)])
            )
            references = [Reference(Output(text=row["translation"]), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=split))

        return instances
