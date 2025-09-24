from typing import List
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
from helm.common.audio_utils import ensure_audio_file_exists_from_array


class UltraSuiteDisorderSymptomsScenario(Scenario):
    """
    A scenario identifying features of speech disorders within the provided audio.
    The audio files contain speech from children, potentially with an adult present.
    """

    name = "speech_disorder"
    description = "A scenario for evaluating speech disorders in children"
    tags = ["audio", "classification", "speech_disorder"]

    def get_instruction(self, words: str) -> str:
        prompt = f"""You are a highly experienced Speech-Language Pathologist (SLP). An audio recording will be provided, typically consisting of a speech prompt from a pathologist followed by a child's repetition. The prompt the child is trying to repeat is as follows: {words}. Based on your professional expertise: 1. Assess the child's speech in the recording and recognize any abnormal features in the child's speech. 2. These features can be on of the following: A - 'substitution', B - 'omission', C - 'addition', D - 'typically_developing', or E - 'stuttering'. Here, 'substitution' is when the child substitutes one word/phrase/syllable for another. 'omission' is when the child omits one word/phrase/syllable. 'addition' is when the child adds one word/phrase/syllable. 'typically_developing' is when the child's speech is typical of a child of their age. 'stuttering' is when the child stutters, has difficulty speaking, repeats sounds/words or prolongs sounds/words. 3. Provide your response as a single letter without any additional explanation, commentary, or unnecessary text."""  # noqa: E501

        return prompt

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Create instances from the audio files and their corresponding JSON annotations.
        The data directory should contain:
        - Audio files (e.g., .mp3)
        - A JSON file with annotations containing 'answer' field
        """
        audio_save_dir = os.path.join(output_path, "audio_files")
        os.makedirs(audio_save_dir, exist_ok=True)

        print("Downloading SAA-Lab/SLPHelmUltraSuitePlus dataset...")
        dataset = load_dataset("SAA-Lab/SLPHelmUltraSuitePlus")

        instances: List[Instance] = []
        split: str = TEST_SPLIT

        for idx, row in enumerate(tqdm(dataset["train"])):
            label = row["disorder_symptom"]
            transcription = row["transcription"]

            unique_id = str(idx)
            local_audio_name = f"{label}_{unique_id}.mp3"
            local_audio_path = os.path.join(audio_save_dir, local_audio_name)
            ensure_audio_file_exists_from_array(local_audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])

            # Create references for each option
            references: List[Reference] = []
            options = ["substitution", "omission", "addition", "typically_developing", "stuttering"]
            if label not in options:
                continue
            for option in options:
                reference = Reference(Output(text=option), tags=[CORRECT_TAG] if option == label else [])
                references.append(reference)

            # Create the input with audio and instruction
            content = [
                MediaObject(content_type="audio/mpeg", location=local_audio_path),
                MediaObject(content_type="text/plain", text=self.get_instruction(transcription)),
            ]

            input = Input(multimedia_content=MultimediaObject(content))
            instances.append(Instance(input=input, references=references, split=split))

        return instances
