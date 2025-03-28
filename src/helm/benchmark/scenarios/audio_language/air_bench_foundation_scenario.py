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
import json


class AirBenchFoundationScenario(Scenario):
    """Air-Bench Foundation

    Air-Bench AIR-Bench (Audio InstRuction Benchmark) is a benchmark designed to evaluate the ability of audio language
    models to understand various types of audio signals (including human speech, natural sounds and music), and
    furthermore, to interact with humans in textual format. AIR-Bench encompasses two dimensions: foundation
    and chat benchmarks. The former consists of 19 tasks with approximately 19k single-choice questions. The
    latter one contains 2k instances of open-ended question-and-answer data. We consider the chat benchmark
    in this scenario.

    Paper: https://aclanthology.org/2024.acl-long.109.pdf
    Code: https://github.com/OFA-Sys/AIR-Bench

    Citation:
    @inproceedings{yang-etal-2024-air,
    title = "{AIR}-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension",
    author = "Yang, Qian  and
      Xu, Jin  and
      Liu, Wenrui  and
      Chu, Yunfei  and
      Jiang, Ziyue  and
      Zhou, Xiaohuan  and
      Leng, Yichong  and
      Lv, Yuanjun  and
      Zhao, Zhou  and
      Zhou, Chang  and
      Zhou, Jingren",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational
        Linguistics (Volume 1: Long Papers)",
    year = "2024",}
    """

    HF_DATA_PATH_PREFIX = "https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset/resolve/main/Foundation"
    META_DATA_FILE_PATH = (
        "https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset/resolve/main/Foundation/Foundation_meta.json"
    )
    SUJECTS_DICT = {
        "music_instrument_classification": "Music_Instruments_Classfication",
        "music_genera_recognition": "Music_Genre_Recognition",
        "music_qa": "Music_AQA",
    }
    OPTION_KEYS = ["choice_a", "choice_b", "choice_c", "choice_d"]

    name = "air_bench_foundation"
    description = "A large-scale dataset of about 46K audio clips to human-written text pairs \
        ([Yang et al, 2024](https://aclanthology.org/2024.acl-long.109.pdf))."
    tags: List[str] = ["audio", "classification", "knowledge"]

    def __init__(self, subject: str) -> None:
        super().__init__()

        if subject not in AirBenchFoundationScenario.SUJECTS_DICT.keys():
            raise ValueError(f"Invalid subject. Valid subjects are: {AirBenchFoundationScenario.SUJECTS_DICT.keys()}")

        self._subject: str = subject

    def _get_subject_indices(self, meta_data) -> List[int]:
        subject_indices = []
        for idx, line in enumerate(meta_data):
            if line["task_name"] == self.SUJECTS_DICT[self._subject]:
                subject_indices.append(idx)
        return subject_indices

    def _get_content_type(self, audio_file_name) -> str:
        if audio_file_name.endswith(".wav"):
            return "audio/wav"
        elif audio_file_name.endswith(".mp3"):
            return "audio/mp3"
        else:
            raise ValueError(f"Unsupported audio file format: {audio_file_name}")

    def _get_label_from_answer(self, row: dict, answer: str):
        for option_key in self.OPTION_KEYS:
            if row[option_key] == answer:
                label = option_key.split("_")[-1].capitalize()
                return label

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        data_dir: str = os.path.join(output_path, "audio_files")
        meta_data_path: str = os.path.join(output_path, "Foundation_meta.json")
        ensure_file_downloaded(source_url=AirBenchFoundationScenario.META_DATA_FILE_PATH, target_path=meta_data_path)
        meta_data = json.load(open(meta_data_path))
        subject_indices = self._get_subject_indices(meta_data)
        valid_testing_indices = []
        for _, row in enumerate(subject_indices):
            audio_meda_data = meta_data[row]
            hf_audio_file_path = os.path.join(
                self.HF_DATA_PATH_PREFIX,
                f'{audio_meda_data["task_name"]}_{audio_meda_data["dataset_name"]}/{audio_meda_data["path"]}',
            )
            local_audio_file_path = os.path.join(
                data_dir, f'{audio_meda_data["task_name"]}_{audio_meda_data["dataset_name"]}_{audio_meda_data["path"]}'
            )
            ensure_file_downloaded(source_url=hf_audio_file_path, target_path=local_audio_file_path)
            if not is_invalid_audio_file(local_audio_file_path):
                valid_testing_indices.append(row)

        for _, row in enumerate(tqdm(valid_testing_indices)):
            audio_meda_data_valid = meta_data[row]
            local_audio_file_path = os.path.join(
                data_dir,
                f'{audio_meda_data_valid["task_name"]}'
                f'_{audio_meda_data_valid["dataset_name"]}_{audio_meda_data_valid["path"]}',
            )

            answer: str = audio_meda_data_valid["answer_gt"]
            references: List[Reference] = []

            answer = self._get_label_from_answer(audio_meda_data_valid, answer)
            # The given correct answer is a letter, but we need an index
            correct_answer_index: int = ord(answer) - ord("A")
            # The options are originally appended to the question

            for i, option_key in enumerate(self.OPTION_KEYS):
                reference: Reference
                is_correct: bool = i == correct_answer_index
                reference = Reference(
                    Output(text=audio_meda_data_valid[option_key]), tags=[CORRECT_TAG] if is_correct else []
                )
                references.append(reference)

            input = Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(
                            content_type=self._get_content_type(audio_meda_data_valid["path"]),
                            location=local_audio_file_path,
                        ),
                        MediaObject(content_type="text/plain", text=audio_meda_data_valid["question"]),
                    ]
                )
            )
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
