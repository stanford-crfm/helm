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


class AirBenchChatScenario(Scenario):
    """Air-Bench Chat

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

    HF_DATA_PATH_PREFIX = "https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset/resolve/main/Chat"
    META_DATA_FILE_PATH = "https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset/resolve/main/Chat/Chat_meta.json"
    SUJECTS = ["music", "sound", "speech", "mix"]

    name = "air_bench_chat"
    description = "A large-scale dataset of about 46K audio clips to human-written text pairs \
        ([Yang et al, 2024](https://aclanthology.org/2024.acl-long.109.pdf))."
    tags: List[str] = ["audio", "reasoning"]

    def __init__(self, subject: str) -> None:
        super().__init__()

        if subject not in AirBenchChatScenario.SUJECTS:
            raise ValueError(f"Invalid subject. Valid subjects are: {AirBenchChatScenario.SUJECTS}")

        self._subject: str = subject

    def _get_subject_indices(self, meta_data) -> List[int]:
        subject_indices = []
        for idx, line in enumerate(meta_data):
            if self._subject == "mix":
                if "_".join(line["task_name"].split("_")[:2]) == "speech_and":
                    subject_indices.append(idx)
            else:
                if line["task_name"].split("_")[0] == self._subject and line["task_name"].split("_")[1] != "and":
                    subject_indices.append(idx)
        return subject_indices

    def _get_content_type(self, audio_file_name) -> str:
        if audio_file_name.endswith(".wav"):
            return "audio/wav"
        elif audio_file_name.endswith(".mp3"):
            return "audio/mp3"
        else:
            raise ValueError(f"Unsupported audio file format: {audio_file_name}")

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        data_dir: str = os.path.join(output_path, "wav_files")
        meta_data_path: str = os.path.join(output_path, "Chat_meta.json")
        ensure_file_downloaded(source_url=AirBenchChatScenario.META_DATA_FILE_PATH, target_path=meta_data_path)
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
            local_audio_file_path_valid = os.path.join(
                data_dir,
                f'{audio_meda_data_valid["task_name"]}'
                f'_{audio_meda_data_valid["dataset_name"]}_{audio_meda_data_valid["path"]}',
            )
            input = Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(
                            content_type=self._get_content_type(audio_meda_data_valid["path"]),
                            location=local_audio_file_path_valid,
                        ),
                        MediaObject(content_type="text/plain", text=audio_meda_data_valid["question"]),
                    ]
                )
            )
            references = [Reference(Output(text=audio_meda_data_valid["answer_gt"]), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
