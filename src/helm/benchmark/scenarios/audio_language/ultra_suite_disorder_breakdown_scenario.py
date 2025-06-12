from typing import List
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
from .ultra_suite_classification_scenario import find_audio_json_pairs


class UltraSuiteDisorderBreakdownScenario(Scenario):
    """
    A scenario for evaluating and classifying specific types of speech disorders in children.
    This scenario extends the basic speech disorder classification by breaking down disorders
    into specific categories: articulation and phonological disorders.
    """

    name = "speech_disorder"
    description = "A scenario for evaluating and classifying specific types of speech disorders in children"
    tags = ["audio", "classification", "speech_disorder", "disorder_breakdown"]
    HF_MAPPING_URL = "https://https://huggingface.co/datasets/SAA-Lab/SLPHelmManualLabels"

    def get_instruction(self, words: str) -> str:
        return f"""You are a highly experienced Speech-Language Pathologist (SLP). An audio recording will be provided, typically consisting of a speech prompt from a pathologist followed by a child's repetition. The prompt text the child is trying to repeat is as follows: {words}. Based on your professional expertise: 1. Assess the child's speech in the recording for signs of typical development or potential speech-language disorder. 2. Conclude your analysis with one of the following labels only: A - 'typically developing' (child's speech patterns and development are within normal age-appropriate ranges), B - 'articulation' (difficulty producing specific speech sounds correctly, such as substituting, omitting, or distorting sounds), C - 'phonological' (difficulty understanding and using the sound system of language, affecting sounds of a particular type). 3. Provide your response as a single letter without any additional explanation, commentary, or unnecessary text."""  # noqa: E501

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Create instances from the audio files and their corresponding JSON annotations.
        The data directory should contain:
        - Audio files (e.g., .mp3)
        - A JSON file with annotations containing 'disorder_class' field
        """
        print(f"Downloading dataset from {UltraSuiteDisorderBreakdownScenario.HF_MAPPING_URL} to {output_path}")
        ensure_file_downloaded(source_url=UltraSuiteDisorderBreakdownScenario.HF_MAPPING_URL, target_path=output_path)

        instances: List[Instance] = []
        split: str = TEST_SPLIT

        # Find all pairs of audio and JSON files
        pairs = find_audio_json_pairs(output_path)
        print(f"Num pairs: {len(pairs)}")

        for audio_path, json_path in tqdm(pairs):
            # Load the annotation
            with open(json_path, "r") as f:
                annotation = json.load(f)

            # Get the correct answer and convert to label
            if "disorder_type" not in annotation or "transcription" not in annotation:
                continue
            label = annotation["disorder_type"]
            prompt = annotation["transcription"]

            # Create references for each option
            references: List[Reference] = []
            for option in ["typically_developing", "articulation", "phonological"]:
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
