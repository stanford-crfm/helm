from typing import List, Tuple
import os
import json

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
from helm.common.general import ensure_file_downloaded


def find_audio_json_pairs(directory: str) -> List[Tuple[str, str]]:
    """
    Find all pairs of MP3 and JSON files in the given directory and its subdirectories.
    Each pair consists of an MP3 file and its corresponding JSON file with the same base name.

    Args:
        directory: Path to the directory containing the files

    Returns:
        List of tuples where each tuple contains (mp3_path, json_path)
    """
    pairs = []

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(directory):
        # Get all MP3 files in current directory
        mp3_files = [f for f in files if f.endswith(".mp3")]

        for mp3_file in mp3_files:
            base_name = os.path.splitext(mp3_file)[0]
            json_file = f"{base_name}.json"

            # Check if corresponding JSON file exists in the same directory
            if json_file in files:
                mp3_path = os.path.join(root, mp3_file)
                json_path = os.path.join(root, json_file)
                pairs.append((mp3_path, json_path))

    return pairs


class UltraSuiteDisorderSymptomsScenario(Scenario):
    """
    A scenario identifying features of speech disorders within the provided audio.
    The audio files contain speech from children, potentially with an adult present.
    """

    name = "speech_disorder"
    description = "A scenario for evaluating speech disorders in children"
    tags = ["audio", "classification", "speech_disorder"]
    HF_MAPPING_URL = "https://https://huggingface.co/datasets/SAA-Lab/SLPHelmManualLabels"

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
        print(f"Downloading dataset from {UltraSuiteDisorderSymptomsScenario.HF_MAPPING_URL} to {output_path}")
        ensure_file_downloaded(source_url=UltraSuiteDisorderSymptomsScenario.HF_MAPPING_URL, target_path=output_path)

        instances: List[Instance] = []
        split: str = TEST_SPLIT

        # Find all pairs of audio and JSON files
        pairs = find_audio_json_pairs(output_path)

        for audio_path, json_path in tqdm(pairs):

            # Load the annotation
            with open(json_path, "r") as f:
                annotation = json.load(f)

            # Get the correct answer and convert to label
            if "disorder_symptom" not in annotation or "transcription" not in annotation:
                continue
            label = annotation["disorder_symptom"]
            prompt = annotation["transcription"]
            # Create references for each option
            references: List[Reference] = []
            for option in ["substitution", "omission", "addition", "typically_developing", "stuttering"]:
                reference = Reference(Output(text=option), tags=[CORRECT_TAG] if option == label else [])
                references.append(reference)

            # Create the input with audio and instruction
            content = [
                MediaObject(content_type="audio/mpeg", location=audio_path),
                MediaObject(content_type="text/plain", text=self.get_instruction(prompt)),
            ]

            input = Input(multimedia_content=MultimediaObject(content))
            instances.append(Instance(input=input, references=references, split=split))

        return instances
