"""Scenarios for audio models"""

from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)
from collections import OrderedDict
from tqdm import tqdm
from datasets import load_dataset
from helm.common.media_object import MediaObject, MultimediaObject


class FLEURSFairnessScenario(Scenario):
    """FLEURS Fairness Scenario

    The FLEURS (Conneau et al, 2022) dataset is an n-way parallel speech dataset in 102 languages
    built on top of the machine translation FLoRes-101 benchmark, with approximately 12 hours of speech
    supervision per language. The task is to identify the language used from the audio sample
    (the Speech Language Identification task).

    Paper: https://arxiv.org/abs/2205.12446
    Code: https://tensorflow.org/datasets/catalog/xtreme_s

    Citation:
    @inproceedings{conneau2023fleurs,
        title={Fleurs: Few-shot learning evaluation of universal representations of speech},
        author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod,
        Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
        booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
        pages={798--805},
        year={2023},
        organization={IEEE}
        }
    """

    HF_DATASET_NAME = "google/xtreme_s"

    GENDERS = {"male": 0, "female": 1}

    name = "fleurs_fairness"
    description = "Language identification for seven languages from seven different language groups \
        ([Conneau et al, 2022](https://arxiv.org/abs/2205.12446))."
    tags: List[str] = ["audio", "recognition", "multilinguality"]

    def __init__(self, gender: str) -> None:
        super().__init__()

        if gender.lower() not in FLEURSFairnessScenario.GENDERS.keys():
            raise ValueError(
                f"Invalid gender input: {gender}. Valid languages are: {FLEURSFairnessScenario.GENDERS.keys()}"
            )

        self._gender: str = gender

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        loading_cases: List[OrderedDict] = []
        overall_dataset = load_dataset(
            FLEURSFairnessScenario.HF_DATASET_NAME,
            name="fleurs.en_us",
            cache_dir=output_path,
            split=TEST_SPLIT,
            trust_remote_code=True,
        )
        for row in tqdm(overall_dataset):
            if row["gender"] == self.GENDERS[self._gender]:
                loading_cases.append(row)
        for row in tqdm(loading_cases):
            local_audio_path = row["path"]
            answer = row["transcription"]
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/wav", location=local_audio_path)])
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="fleurs_fairness",
            display_name="FLEURS Fairness",
            description="FLEURS is an n-way parallel speech dataset in 102 languages built on top of "
            "the machine translation FLoRes-101  benchmark, with approximately 12 hours of "
            "speech supervision per language. FLEURS can be used for a variety of  speech "
            "tasks, including Automatic Speech Recognition (ASR), Speech Language "
            "Identification (Speech LangID),  Translation and Retrieval.\n"
            "We only use the English subset of the dataset for the fairness task. We ask "
            "the model to do ASR on audio files from different gender groups ([Conneau et "
            "al, 2022](https://arxiv.org/abs/2205.12446)).\n",
            taxonomy=TaxonomyInfo(
                task="audio classification",
                what="audio, transcripts, and gender of the speaker",
                when="2022",
                who="real speakers",
                language="English",
            ),
            main_metric="wer_score",
            main_split="test",
        )
