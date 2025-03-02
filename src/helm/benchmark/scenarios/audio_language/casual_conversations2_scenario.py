from typing import List, Optional
import os

from tqdm import tqdm
import json

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
from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.audio_utils import extract_audio


class CasualConversations2Scenario(Scenario):
    """
    Casual Conversation v2 (Porgali et al, 2023) is composed of over 5,567 participants (26,467 videos).
    The videos feature paid individuals who agreed to participate in the project and explicitly provided
    Age, Gender, Language/Dialect, Geo-location, Disability, Physical adornments, Physical attributes labels
    themselves. The videos were recorded in Brazil, India, Indonesia, Mexico, Philippines, United States,
    and Vietnam with a diverse set of adults in various categories.

    The dataset contains the audio, speaker's age, gender information in the following languages:
    English, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Tamil, Telugu, and Vietnamese.

    Paper: https://arxiv.org/abs/2303.04838
    Dataset: https://ai.meta.com/datasets/casual-conversations-v2-dataset/

    Requires downloading Causal Conversations V2 from https://ai.meta.com/datasets/casual-conversations-v2-downloads

    Citation:
    @inproceedings{porgali2023casual,
    title={The casual conversations v2 dataset},
    author={Porgali, Bilal and Albiero, V{\'\i}tor and Ryda, Jordan and Ferrer, Cristian Canton and Hazirbas, Caner},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={10--17},
    year={2023}
    }
    """

    SUBJECTS = ["age", "gender"]
    SCRIPT_DOWNLOADING_URL = (
        "https://huggingface.co/datasets/UCSC-VLAA/Causal_Conversation_V2_script/"
        "resolve/main/CasualConversationsV2_v2.json"
    )
    AGE_INSTRUCTION = "Listen to the audio and take your best guess to estimate the speaker's age."
    GENDER_INSTRUCTION = "Listen to the audio and take your best guess to determine the speaker's gender."
    name = "casual_conversations2"
    description = (
        "A large scale multilingual speech translation corpus "
        "([Porgali et al., 2023](https://arxiv.org/abs/2303.04838))."
    )
    tags = ["audio", "classification", "multilinguality"]
    gender_options: List[str] = ["male", "female", "transgender male", "transgender female", "non-binary", "other"]
    age_options: List[str] = ["18-30", "31-50", "51+", "other"]

    def __init__(self, subject: str) -> None:
        super().__init__()

        if subject not in self.SUBJECTS:
            raise ValueError(f"Invalid subject. Valid subjects are: {CasualConversations2Scenario.SUBJECTS}")

        self._subject: str = subject
        self._convert_answer_to_label_func = (
            self._convert_age_to_label if subject == "age" else self._convert_gender_to_label
        )
        self.options = self.age_options if subject == "age" else self.gender_options
        self.instruction = self.AGE_INSTRUCTION if subject == "age" else self.GENDER_INSTRUCTION

    def _convert_age_to_label(self, age: str) -> str:
        if age != "prefer not to say":
            age_int = int(age)
            if 18 <= age_int <= 30:
                return "A"
            elif 31 <= age_int <= 50:
                return "B"
            elif 51 <= age_int:
                return "C"
            else:
                raise ValueError(f"Invalid age: {age}")
        else:
            return "D"

    def _convert_gender_to_label(self, gender: Optional[str]) -> str:
        if gender is not None and gender != "prefer not to say":
            if gender == "cis man":
                return "A"
            elif gender == "cis woman":
                return "B"
            elif gender == "transgender man":
                return "C"
            elif gender == "transgender woman":
                return "D"
            elif gender == "non-binary":
                return "E"
            else:
                raise ValueError(f"Invalid gender: {gender}")
        else:
            return "F"

    def get_instances(self, output_path: str) -> List[Instance]:
        data_dir: str = os.path.join(output_path, "videos_files")
        assert os.path.exists(data_dir), (
            f"Download the video files from Meta's Casual Conversations v2 dataset from "
            f"(https://ai.meta.com/datasets/casual-conversations-v2-downloads) and unzip and place at {data_dir}."
        )
        script_file_path: str = os.path.join(output_path, "CasualConversationsV2.json")
        audio_file_folder: str = os.path.join(output_path, "audio_files")
        ensure_directory_exists(audio_file_folder)
        ensure_file_downloaded(self.SCRIPT_DOWNLOADING_URL, script_file_path)
        audio_scripts = json.load(open(script_file_path))

        instances: List[Instance] = []
        split: str = TEST_SPLIT

        for file_name in tqdm(os.listdir(data_dir)):
            if file_name.endswith(".mp4"):
                local_audio_path: str = os.path.join(audio_file_folder, file_name.replace(".mp4", ".mp3"))
                local_video_path: str = os.path.join(data_dir, file_name)

                if not os.path.exists(local_audio_path):
                    extract_audio(local_video_path, local_audio_path)
                assert os.path.exists(local_audio_path), f"Audio file does not exist at path: {local_audio_path}"

                subject_answer = audio_scripts[file_name][self._subject]
                answer = self._convert_answer_to_label_func(subject_answer)
                # The given correct answer is a letter, but we need an index
                correct_answer_index: int = ord(answer) - ord("A")
                # The options are originally appended to the question

                references: List[Reference] = []
                for i, option in enumerate(self.options):
                    reference: Reference
                    is_correct: bool = i == correct_answer_index
                    reference = Reference(Output(text=option), tags=[CORRECT_TAG] if is_correct else [])
                    references.append(reference)

                content = [
                    MediaObject(content_type="audio/mpeg", location=local_audio_path),
                    MediaObject(content_type="text/plain", text=self.instruction),
                ]

                input = Input(multimedia_content=MultimediaObject(content))
                instances.append(Instance(input=input, references=references, split=split))

        return instances
