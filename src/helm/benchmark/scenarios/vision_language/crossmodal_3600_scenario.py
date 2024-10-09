import json
import os
from typing import Dict, List

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded


class Crossmodal3600Scenario(Scenario):
    """
    Crossmodal-3600 dataset (XM3600 in short), a geographically-diverse set of 3600 images annotated
    with human-generated reference captions in 36 languages.

    @inproceedings{ThapliyalCrossmodal2022,
      author        = {Ashish Thapliyal and Jordi Pont-Tuset and Xi Chen and Radu Soricut},
      title         = {{Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset}},
      booktitle     = {EMNLP},
      year          = {2022}
    }

    Paper: https://arxiv.org/abs/2205.12522
    Website: https://google.github.io/crossmodal-3600/
    """

    LANGUAGE_TO_ID: Dict[str, str] = {
        "arabic": "ar",
        "bengali": "bn",
        "chinese": "zh",
        "croatian": "hr",
        "cusco_quechua": "quz",
        "czech": "cs",
        "danish": "da",
        "dutch": "nl",
        "english": "en",
        "persian": "fa",
        "finnish": "fi",
        "french": "fr",
        "german": "de",
        "greek": "el",
        "hebrew": "he",
        "hindi": "hi",
        "hungarian": "hu",
        "indonesian": "id",
        "italian": "it",
        "japanese": "ja",
        "korean": "ko",
        "maori": "mi",
        "norwegian": "no",
        "polish": "pl",
        "portuguese": "pt",
        "romanian": "ro",
        "russian": "ru",
        "spanish": "es",
        "swahili": "sw",
        "swedish": "sv",
        "telugu": "te",
        "thai": "th",
        "turkish": "tr",
        "ukrainian": "uk",
        "vietnamese": "vi",
    }

    IMAGES_URL: str = "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz"
    CAPTIONS_URL: str = "https://google.github.io/crossmodal-3600/web-data/captions.zip"

    name = "crossmodal_3600"
    description = (
        "Crossmodal-3600 dataset (XM3600 in short), a geographically-diverse set of 3600 images annotated "
        "with human-generated reference captions in 36 languages. "
        "([Thapliyal et al., 2022)](https://arxiv.org/abs/2205.12522))."
    )
    tags = ["vision-language", "multilinguality"]

    def __init__(self, location: str, language: str):
        super().__init__()
        self._locale_id: str = self.LANGUAGE_TO_ID[location]
        self._language_id: str = self.LANGUAGE_TO_ID[language]
        self._instruction: str = f"Generate a short caption for the following image in {language}."

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_file_downloaded(
            source_url=self.IMAGES_URL,
            target_path=images_path,
            unpack=True,
            unpack_type="untar",
        )

        captions_path: str = os.path.join(output_path, "captions.jsonl")
        ensure_file_downloaded(
            source_url=self.CAPTIONS_URL,
            target_path=captions_path,
            unpack=True,
            unpack_type="unzip",
        )

        instances: List[Instance] = []
        with open(captions_path, "r") as captions_file:
            for line in captions_file:
                example: Dict = json.loads(line)

                locale_id: str = example["image/locale"]
                if locale_id != self._locale_id:
                    continue

                key: str = example["image/key"]
                image_path: str = os.path.join(images_path, f"{key}.jpg")
                assert os.path.exists(image_path), f"Image {image_path} does not exist"

                assert self._language_id in example, f"Language {self._language_id} not found in example"
                all_captions: Dict = example[self._language_id]
                captions: List[str] = all_captions["caption"]

                content: List[MediaObject] = [
                    MediaObject(text=self._instruction, content_type="text/plain"),
                    MediaObject(location=image_path, content_type="image/jpeg"),
                ]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=[Reference(Output(text=caption), tags=[CORRECT_TAG]) for caption in captions],
                        split=TEST_SPLIT,
                    )
                )

        return instances
