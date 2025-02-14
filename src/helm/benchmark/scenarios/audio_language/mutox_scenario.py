import os
import pdb
import requests
from io import BytesIO
from typing import List

from pydub import AudioSegment
from tqdm import tqdm
import pandas as pd

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.audio_utils import is_invalid_audio_file
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.common.hierarchical_logger import hlog


class MuToxScenario(Scenario):
    """
    MuTox: MuTox: Universal MUltilingual Audio-based TOXicity Dataset and Zero-shot Detector

    MuTox, the first highly multilingual audio-based dataset with toxicity labels. The dataset consists of 20k
    audio utterances for English and Spanish, and 4k for the other 19 languages. To showcase the quality of this
    dataset, we train the MuTox audio-based toxicity classifier, which allows zero-shot toxicity detection across
    a broad range of languages. This classifier outperforms existing text-based trainable classifiers by more than
    1% AUC, while increasing the language coverage from 8 to 100+ languages. When compared to a wordlist-based
    classifier that covers a similar number of languages, MuTox improves precision and recall by ∼2.5 times.

    Languages:
        English
        Spanish

        Arabic
        Bengali
        Mandarin Chinese
        Dutch
        French
        German
        Hindi
        Indonesian
        Italian
        Japanese
        Korean
        Portuguese
        Russian
        Swahili
        Tagalog
        Thai
        Turkish
        Urdu
        Vietnamese

    The columns of the dataset are:

    id: a string id of the segment;
    lang: 3-letter language code;
    partition: one of train, dev, or devtest
    public_url_segment: a string formatted as url:start:end, where start and end are indicated in milliseconds;
    audio_file_transcript: text transctiption of the segment;
    contains_toxicity, toxicity_types, perlocutionary_effects: annotation results as strings
    label: an integer label, equal to 1 if contains_toxicity equals Yes and 0 otherwise;
    etox_result: toxic word (or multiple words, separated by |) detected by the Etox matcher;
    detoxify_score: toxicity probabilities predicted by the Detoxify system (float numbers between 0 and 1);
    mutox_speech_score, mutox_text_score, mutox_zero_shot_speech_score, mutox_zero_shot_text_score: MuTox predictions
    as float numbers with any value (they can be interpreted as logits,
    i.e. probabilities before a sigmoid transformation).

    Citation:

    @misc{costajussà2023mutox,
      title={MuTox: Universal MUltilingual Audio-based TOXicity Dataset and Zero-shot Detector},
      author={ Marta R. Costa-jussà, Mariano Coria Meglioli, Pierre Andrews, David Dale, Prangthip Hansanti,
      Elahe Kalbassi, Alex Mourachko, Christophe Ropers, Carleigh Wood},
      year={2023},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
    """  # noqa: E501

    ANNOTATIONS_URL = "https://dl.fbaipublicfiles.com/seamless/datasets/mutox.tsv"

    LANGAUGE_CODES = {
        "English": "eng",
        "Spanish": "spa",
        "Arabic": "ara",
        "Bengali": "ben",
        "Mandarin Chinese": "zho",
        "Dutch": "nld",
        "French": "fra",
        "German": "deu",
        "Hindi": "hin",
        "Indonesian": "ind",
        "Italian": "ita",
        "Japanese": "jpn",
        "Korean": "kor",
        "Portuguese": "por",
        "Russian": "rus",
        "Swahili": "swa",
        "Tagalog": "tgl",
        "Thai": "tha",
        "Turkish": "tur",
        "Urdu": "urd",
        "Vietnamese": "vie",
    }

    name = "mutox"
    description = "Toxicity detection benchmark ([Costa-jussà et al, 2018](https://arxiv.org/abs/2401.05060))."
    tags = ["audio", "classification", "toxicity "]

    def __init__(self, language: str) -> None:
        super().__init__()
        self._language_code: str = self.LANGAUGE_CODES[language]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the annotations
        annotations_path: str = os.path.join(output_path, "mutox.tsv")
        ensure_file_downloaded(self.ANNOTATIONS_URL, annotations_path)

        # Where the audio files will be downloaded to
        audio_path: str = os.path.join(output_path, "audio")
        ensure_directory_exists(audio_path)

        instances: List[Instance] = []
        df = pd.read_csv(annotations_path, delimiter="\t")
        for row in tqdm(df.itertuples()):
            if row.partition != "devtest":
                continue

            if row.lang != self._language_code:
                continue

            local_audio_path: str = os.path.join(audio_path, f"{row.id}.mp3")
            if not os.path.exists(local_audio_path):
                # The provided URL has the complete audio, so we need to download it and clip it
                # public_url_segment: a string formatted as url:start:end,
                parts = row.public_url_segment.split()
                assert len(parts) == 3, f"Invalid public_url_segment format: {row}"
                audio_url, start_ms, end_ms = parts
                start_ms, end_ms = int(start_ms), int(end_ms)

                # Download the full audio file
                try:
                    response = requests.get(audio_url, stream=True)
                    response.raise_for_status()

                    # Load audio from the downloaded file
                    audio = AudioSegment.from_file(BytesIO(response.content))

                    # Clip the audio
                    clipped_audio = audio[start_ms:end_ms]

                    # Save the clipped file
                    clipped_audio.export(local_audio_path, format="mp3")
                except Exception as e:
                    # Some files are no longer available or invalid
                    hlog(f"Skipping this example...error downloading and processing audio file from {audio_url}: {e}")
                    continue

            if is_invalid_audio_file(local_audio_path):
                hlog(f"Skipping this example. Invalid audio file at {local_audio_path}")
                continue

            input = Input(
                multimedia_content=MultimediaObject(
                    media_objects=[
                        # Following what was prompted annotators as outlined in the paper
                        MediaObject(text="Does the utterance contain toxicity?", content_type="text/plain"),
                        MediaObject(location=local_audio_path, content_type="audio/mp3"),
                    ]
                )
            )
            references = [
                Reference(Output(text="Yes"), tags=[CORRECT_TAG] if row.label == 1 else []),
                Reference(Output(text="No"), tags=[CORRECT_TAG] if row.label == 0 else []),
            ]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))

        assert len(instances) > 0, f"No instances found for language: {self._language_code}"
        return instances
