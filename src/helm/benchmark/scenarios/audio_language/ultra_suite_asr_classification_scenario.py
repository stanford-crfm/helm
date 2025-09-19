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


class UltraSuiteASRClassificationScenario(Scenario):
    """
    A scenario for evaluating whether a child speaker has a speech disorder or not.
    The audio files contain speech from children, potentially with an adult present.
    The task is to classify whether the child speaker is typically developing or has a speech disorder.
    """

    name = "speech_disorder"
    description = "A scenario for evaluating speech disorders in children"
    tags = ["audio", "classification", "speech_disorder", "asr"]

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

            label = row["disorder_class"]
            transcription = row["transcription"]

            unique_id = str(idx)
            local_audio_name = f"{label}_{unique_id}.mp3"
            local_audio_path = os.path.join(audio_save_dir, local_audio_name)
            ensure_audio_file_exists_from_array(local_audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])

            # Create references for each option
            references: List[Reference] = []
            for option in ["typically_developing", "speech_disorder"]:
                reference = Reference(Output(text=option), tags=[CORRECT_TAG] if option == label else [])
                references.append(reference)

            # Create the input with audio and instruction
            content = [
                MediaObject(content_type="audio/mpeg", location=local_audio_path),
            ]

            input = Input(multimedia_content=MultimediaObject(content))
            instances.append(
                Instance(input=input, references=references, split=split, extra_data={"transcription": transcription})
            )

        return instances
