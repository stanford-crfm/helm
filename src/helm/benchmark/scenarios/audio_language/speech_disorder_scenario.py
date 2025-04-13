from typing import List
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


class SpeechDisorderScenario(Scenario):
    """
    A scenario for evaluating whether a child speaker has a speech disorder or not.
    The audio files contain speech from children, potentially with an adult present.
    The task is to classify whether the child speaker is typically developing or has a speech disorder.
    """

    name = "speech_disorder"
    description = "A scenario for evaluating speech disorders in children"
    tags = ["audio", "classification", "speech_disorder"]

    # Classification options
    options: List[str] = ["typically developing", "has a speech disorder"]

    def __init__(self, subject: str) -> None:
        super().__init__()
        self.subject = subject
    
    def get_instruction(self, words: str, age: str, gender: str) -> str:
        return f"Listen to the audio and determine if the child speaker has a speech disorder or is typically developing. The speaker is a {age}-year-old {gender} trying to say {words}."

    def _convert_answer_to_label(self, answer: str) -> str:
        """Convert the answer from the JSON to a label (A or B)"""
        if answer == "typically developing":
            return "A"
        elif answer == "has a speech disorder":
            return "B"
        else:
            raise ValueError(f"Invalid answer: {answer}")

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Create instances from the audio files and their corresponding JSON annotations.
        The data directory should contain:
        - Audio files (e.g., .mp3, .wav)
        - A JSON file with annotations containing 'answer' field
        """
        ensure_directory_exists(output_path)
        
        instances: List[Instance] = []
        split: str = TEST_SPLIT

        # Get all audio files in the directory
        audio_files = [f for f in os.listdir(output_path) if f.endswith(('.mp3'))]
        
        for audio_file in tqdm(audio_files):
            # Construct paths
            audio_path = os.path.join(output_path, audio_file)
            json_path = os.path.join(output_path, f"{os.path.splitext(audio_file)[0]}.json")
            text_path = os.path.join(output_path, f"{os.path.splitext(audio_file)[0]}.txt")

            # Load the text
            with open(text_path, 'r') as f:
                lines = f.readlines()
                words = lines[0].strip()  # First line: words
                age_gender = lines[2].strip()  # Third line: age and gender
                # Decode age and gender (format: 01M)
                age = age_gender[:2]  # First two digits are age
                gender = age_gender[2]  # Last character is gender (M/F)
            
            
            
            # Load the annotation
            with open(json_path, 'r') as f:
                annotation = json.load(f)
            
            # Get the correct answer and convert to label
            answer = annotation['answer']
            label = self._convert_answer_to_label(answer)
            
            # Create references for each option
            references: List[Reference] = []
            for i, option in enumerate(self.options):
                is_correct = i == (ord(label) - ord('A'))
                reference = Reference(Output(text=option), tags=[CORRECT_TAG] if is_correct else [])
                references.append(reference)
            
            # Create the input with audio and instruction
            content = [
                MediaObject(content_type="audio/mpeg", location=audio_path),
                MediaObject(content_type="text/plain", text=self.get_instruction(words, age, gender)),
            ]
            
            input = Input(multimedia_content=MultimediaObject(content))
            instances.append(Instance(input=input, references=references, split=split))
        
        return instances 