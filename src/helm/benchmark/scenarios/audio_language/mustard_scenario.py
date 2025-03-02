import json
import os
from typing import List

from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.audio_utils import is_invalid_audio_file, extract_audio
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class MUStARDScenario(Scenario):
    """
    MUStARD: Multimodal Sarcasm Detection Dataset

    A multimodal video corpus for research in automated sarcasm discovery. The dataset is compiled from popular
    TV shows including Friends, The Golden Girls, The Big Bang Theory, and Sarcasmaholics Anonymous. MUStARD consists
    of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, providing
    additional information on the scenario where it occurs.

    We just extract the audio from the given videos.

    The columns of the dataset are:
    - utterance: The text of the target utterance to classify.
    - speaker: Speaker of the target utterance.
    - context: List of utterances (in chronological order) preceding the target utterance.
    - context_speakers: Respective speakers of the context utterances.
    - sarcasm: Binary label for sarcasm tag.

    More specifically an example looks like this:

    "1_60": {
        "utterance": "It's just a privilege to watch your mind at work.",
        "speaker": "SHELDON",
        "context": [
            "I never would have identified the fingerprints of string theory in the aftermath of the Big Bang.",
            "My apologies. What's your plan?"
        ],
        "context_speakers": [
            "LEONARD",
            "SHELDON"
        ],
        "show": "BBT",
        "sarcasm": true
    }

    The key is the video id.

    The video folder has two subfolders:
    - context_final: Contains the context videos (e.g., 1_60_c.mp4)
    - utterances_final: Contains the target utterance videos (e.g., 1_60.mp4)

    Citation:

    @inproceedings{mustard,
        title = "Towards Multimodal Sarcasm Detection (An \_Obviously\_ Perfect Paper)",
        author = "Castro, Santiago  and
          Hazarika, Devamanyu  and
          P{\'e}rez-Rosas, Ver{\'o}nica  and
          Zimmermann, Roger  and
          Mihalcea, Rada  and
          Poria, Soujanya",
        booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics
                    (Volume 1: Long Papers)",
        month = "7",
        year = "2019",
        address = "Florence, Italy",
        publisher = "Association for Computational Linguistics",
    }
    """

    RAW_VIDEO_CLIPS_URL: str = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
    ANNOTATIONS_URL: str = (
        "https://raw.githubusercontent.com/soujanyaporia/MUStARD/refs/heads/master/data/" "sarcasm_data.json"
    )

    name = "mustard"
    description = "Sarcasm detection benchmark ([Castro et al, 2018](https://arxiv.org/abs/1906.01815))."
    tags = ["audio", "classification", "toxicity", "sarcasm detection"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the annotations
        annotations_path: str = os.path.join(output_path, "sarcasm_data.json")
        ensure_file_downloaded(self.ANNOTATIONS_URL, annotations_path)

        # Where the video files will be downloaded to
        video_path: str = os.path.join(output_path, "video")
        ensure_file_downloaded(self.RAW_VIDEO_CLIPS_URL, video_path, unpack=True)

        # Where the audio files will be extracted to
        audio_path: str = os.path.join(output_path, "audio")
        ensure_directory_exists(audio_path)

        instances: List[Instance] = []
        annotations = json.load(open(annotations_path, "r"))
        for key, row in tqdm(annotations.items()):
            # Extract the audio from the context video
            context_audio_path: str = os.path.join(audio_path, f"{key}_c.mp3")
            if not os.path.exists(context_audio_path):
                # Extract the audio from the video
                context_video_path: str = os.path.join(video_path, "context_final", f"{key}_c.mp4")
                extract_audio(context_video_path, context_audio_path)
            assert not is_invalid_audio_file(context_audio_path), f"Invalid audio file: {context_audio_path}"

            # Extract the audio from the target utterance video
            utterance_audio_path: str = os.path.join(audio_path, f"{key}.mp3")
            if not os.path.exists(utterance_audio_path):
                utterance_video_path: str = os.path.join(video_path, "utterances_final", f"{key}.mp4")
                extract_audio(utterance_video_path, utterance_audio_path)
            assert not is_invalid_audio_file(utterance_audio_path), f"Invalid audio file: {utterance_audio_path}"

            input = Input(
                multimedia_content=MultimediaObject(
                    media_objects=[
                        # Input both the context and the utterance audio
                        MediaObject(text="Context:", content_type="text/plain"),
                        MediaObject(location=context_audio_path, content_type="audio/mpeg"),
                        MediaObject(text="Utterance:", content_type="text/plain"),
                        MediaObject(location=utterance_audio_path, content_type="audio/mpeg"),
                        MediaObject(
                            text="Given the context, does the utterance contain sarcasm?", content_type="text/plain"
                        ),
                    ]
                )
            )
            is_sarcastic: bool = row["sarcasm"]
            references = [
                Reference(Output(text="Yes"), tags=[CORRECT_TAG] if is_sarcastic else []),
                Reference(Output(text="No"), tags=[CORRECT_TAG] if not is_sarcastic else []),
            ]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))

        return instances
