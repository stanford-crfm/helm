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
import json


class AudioPAIRSScenario(Scenario):
    """Audio PAIRS

    Audio PAIRS is an audio extension of the PAIRS dataset (Fraser et al, 2024) to examine gender and
    racial bias in audio large language models. We convert the questions in the PAIRS dataset to audio
    clips using OpenAI's TTS-1-HD API.

    This dataset is also modified to add an option to opt-out with "unclear" as a choice.
    """

    DOWNLOADING_URL = "https://huggingface.co/datasets/UCSC-VLAA/Audio_PAIRS/resolve/main/audio_pairs_files.zip"
    SUJECTS = ["occupation", "status", "potential_crime"]

    name = "audio_pairs"
    description = "Examining gender and racial bias in AudioLMs using a converted audio from the PAIRS dataset."
    tags: List[str] = ["audio", "classification"]

    def __init__(self, subject: str) -> None:
        super().__init__()

        if subject not in AudioPAIRSScenario.SUJECTS:
            raise ValueError(f"Invalid subject. Valid subjects are: {AudioPAIRSScenario.SUJECTS}")

        self._subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        downloading_dir: str = os.path.join(output_path, "download")
        ensure_file_downloaded(source_url=AudioPAIRSScenario.DOWNLOADING_URL, target_path=downloading_dir, unpack=True)
        data_dir: str = os.path.join(downloading_dir, "audio_pairs_files")
        audio_file_folder = os.path.join(data_dir, self._subject)
        audio_instruction_path = os.path.join(data_dir, "audio_pairs_instructions.json")
        audio_instructions = json.load(open(audio_instruction_path))[self._subject]
        for audio_file_name, instruction in tqdm(audio_instructions.items()):
            local_audio_file_name = "_".join(audio_file_name.split("_")[:-1]) + ".mp3"
            local_audio_path: str = os.path.join(audio_file_folder, local_audio_file_name)
            content = [
                MediaObject(content_type="audio/mpeg", location=local_audio_path),
                MediaObject(content_type="text/plain", text=instruction),
            ]
            input = Input(multimedia_content=MultimediaObject(content))
            references = [Reference(Output(text="unclear"), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
