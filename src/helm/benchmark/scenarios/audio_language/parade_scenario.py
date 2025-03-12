from typing import List
import os
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
from tqdm import tqdm
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded


class PARADEScenario(Scenario):
    """PARADE

    PARADE dataset is inspired by the PAIRS dataset for evaluating occupation and status bias
    in vision-language models. We collect a new dataset of audio-text multi-choice QA task that
    involves exploring occupation and status bias. The dataset consists of 436 audio-text QA pairs
    with 3 options each.
    """

    ANNOT_URL = (
        "https://huggingface.co/datasets/UCSC-VLAA/PARADE_audio/resolve/main/audio_result" "_path_mapping_v2.json"
    )
    MALE_AUDIO = "https://huggingface.co/datasets/UCSC-VLAA/PARADE_audio/resolve/main/onyx.zip"
    FEMALE_AUDIO = "https://huggingface.co/datasets/UCSC-VLAA/PARADE_audio/resolve/main/nova.zip"

    PARADE_INSTRUCTION = "\n\n Answer the question with one of the following options: A, B, or C."

    SUBSET_LIST = ["occupation", "status"]
    VOICE_MAPPING = {"male": "onyx", "female": "nova"}

    name = "parade"
    description = "Exploring occupation and status bias in the audio-text multi-choice QA task."
    tags: List[str] = ["audio", "bias"]

    def __init__(self, subset: str, voice: str) -> None:
        super().__init__()

        subset = subset.lower()
        voice = voice.lower()
        if subset not in PARADEScenario.SUBSET_LIST:
            raise ValueError(f"Invalid subset. Valid subsets are: {PARADEScenario.SUBSET_LIST}")

        if voice not in PARADEScenario.VOICE_MAPPING.keys():
            raise ValueError(f"Invalid voice. Valid voices are: {PARADEScenario.VOICE_MAPPING.keys()}")

        self._subset: str = subset
        self._voice: str = voice

    def _convert_answer_to_label(self, options: list, answer: str) -> str:
        option_list = ["A", "B", "C"]
        return option_list[options.index(answer)]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        annot_save_path = os.path.join(output_path, "annotation.json")
        ensure_file_downloaded(source_url=PARADEScenario.ANNOT_URL, target_path=annot_save_path)
        annotations = json.load(open(annot_save_path))[self.VOICE_MAPPING[self._voice]][self._subset]
        test_annotations = []
        for key in annotations:
            for key2 in annotations[key]:
                test_annotations.append(annotations[key][key2])
        audio_save_dir = os.path.join(output_path, "audio_files")
        if self._voice == "male":
            ensure_file_downloaded(source_url=PARADEScenario.MALE_AUDIO, target_path=audio_save_dir, unpack=True)
        else:
            ensure_file_downloaded(source_url=PARADEScenario.FEMALE_AUDIO, target_path=audio_save_dir, unpack=True)
        for row in tqdm(test_annotations):
            local_audio_path = os.path.join(output_path, "audio_files", row["path"])
            answer = self._convert_answer_to_label(row["options"], row["label"])
            # The given correct answer is a letter, but we need an index
            correct_answer_index: int = ord(answer) - ord("A")
            references: List[Reference] = []
            question = row["question"]
            for i, option in enumerate(row["options"]):
                reference: Reference
                is_correct: bool = i == correct_answer_index
                reference = Reference(Output(text=option), tags=[CORRECT_TAG] if is_correct else [])
                references.append(reference)

            input = Input(
                multimedia_content=MultimediaObject(
                    [
                        MediaObject(content_type="audio/mpeg", location=local_audio_path),
                        MediaObject(content_type="text/plain", text=question + self.PARADE_INSTRUCTION),
                    ]
                )
            )
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
