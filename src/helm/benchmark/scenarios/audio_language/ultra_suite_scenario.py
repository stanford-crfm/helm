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
from helm.common.general import ensure_directory_exists


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


class UltraSuiteScenario(Scenario):
    """
    A scenario for evaluating whether a child speaker has a speech disorder or not.
    The audio files contain speech from children, potentially with an adult present.
    The task is to classify whether the child speaker is typically developing or has a speech disorder.
    You can find the dataset at https://ultrasuite.github.io/. The base dataset is pre-processed to do the following:
    1. Convert the audio to MP3 format
    2. Build a JSON file with the following format:
    {
        "words": ["word1", "word2", "word3"],
        "answer": "typically developing" or "speech disorder"
    }
    where "words" is a list of words that the child is expected to say and "answer" is the correct label.
    The word ground truth is derived from a .txt file associated with each audio file.
    """

    name = "speech_disorder"
    description = "A scenario for evaluating speech disorders in children"
    tags = ["audio", "classification", "speech_disorder"]

    def get_instruction(self, words: str) -> str:
        return f"""You are a highly experienced Speech-Language Pathologist (SLP). 
An audio recording will be provided, typically consisting of a speech prompt 
from a pathologist followed by a child's repetition. 
Based on your professional expertise:

1. Assess the child's speech in the recording for signs of typical development 
or potential speech-language disorder.
2. Conclude your analysis with one of the following labels only: 
'Typically developing' or 'Speech disorder'.
3. Provide your response without any additional explanation, commentary, 
or unnecessary text. Only 'Typically developing' or 'Speech disorder'.

The prompt text and the utterance recording date/time are as follows: {words}"""

    def _convert_answer_to_label(self, answer: str) -> str:
        """Convert the answer from the JSON to a label (A or B)"""
        answer = answer.lower()
        if answer == "typically developing":
            return "A"
        elif answer == "speech disorder":
            return "B"
        else:
            raise ValueError(f"Invalid answer: {answer}")

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Create instances from the audio files and their corresponding JSON annotations.
        The data directory should contain:
        - Audio files (e.g., .mp3)
        - A JSON file with annotations containing 'answer' field
        """
        ensure_directory_exists(output_path)

        instances: List[Instance] = []
        split: str = TEST_SPLIT

        # Find all pairs of audio and JSON files
        pairs = find_audio_json_pairs(output_path)

        for audio_path, json_path in tqdm(pairs):

            # Load the annotation
            with open(json_path, "r") as f:
                annotation = json.load(f)

            # Get the correct answer and convert to label
            answer = annotation["answer"]
            print(f"Answer: {answer}, path: {audio_path}")
            words = " ".join(annotation["words"])
            label = self._convert_answer_to_label(answer)
            print(f"Label: {label}")
            # Create references for each option
            references: List[Reference] = []
            reference = Reference(Output(text=label), tags=[CORRECT_TAG])
            references.append(reference)

            # Create the input with audio and instruction
            content = [
                MediaObject(content_type="audio/mpeg", location=audio_path),
                MediaObject(content_type="text/plain", text=self.get_instruction(words)),
            ]

            input = Input(multimedia_content=MultimediaObject(content))
            instances.append(Instance(input=input, references=references, split=split))

        return instances
