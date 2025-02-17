from io import BytesIO
from typing import List
import os
import requests

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
from helm.common.hierarchical_logger import hlog, htrack_block


class MuToxScenario(Scenario):
    """
    MuTox: MuTox: Universal MUltilingual Audio-based TOXicity Dataset and Zero-shot Detector

    MuTox, the first highly multilingual audio-based dataset with toxicity labels. The dataset consists of 20k
    audio utterances for English and Spanish, and 4k for the other languages. To showcase the quality of this
    dataset, we train the MuTox audio-based toxicity classifier, which allows zero-shot toxicity detection across
    a broad range of languages. This classifier outperforms existing text-based trainable classifiers by more than
    1% AUC, while increasing the language coverage from 8 to 100+ languages. When compared to a wordlist-based
    classifier that covers a similar number of languages, MuTox improves precision and recall by ∼2.5 times.

    Languages:
        "Arabic": "arb",
        "Bengali": "ben",
        "Bulgarian": "bul",
        "Catalan": "cat",
        "Czech": "ces",
        "Mandarin Chinese": "cmn",
        "Danish": "dan",
        "German": "deu",
        "Greek": "ell",
        "English": "eng",
        "Estonian": "est",
        "Western Persian": "fas",
        "Finnish": "fin",
        "French": "fra",
        "Hebrew": "heb",
        "Hindi": "hin",
        "Hungarian": "hun",
        "Indonesian": "ind",
        "Italian": "ita",
        "Dutch": "nld",
        "Polish": "pol",
        "Portuguese": "por",
        "Russian": "rus",
        "Spanish": "spa",
        "Slovak": "slk",
        "Swahili": "swh",
        "Tagalog": "tgl",
        "Turkish": "tur",
        "Urdu": "urd",
        "Vietnamese": "vie",

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
    """

    ANNOTATIONS_URL = "https://dl.fbaipublicfiles.com/seamless/datasets/mutox.tsv"

    LANGAUGE_CODES = {
        "Arabic": "arb",
        "Bengali": "ben",
        "Bulgarian": "bul",
        "Catalan": "cat",
        "Czech": "ces",
        "Mandarin_Chinese": "cmn",
        "Danish": "dan",
        "German": "deu",
        "Greek": "ell",
        "English": "eng",
        "Estonian": "est",
        "Western_Persian": "fas",
        "Finnish": "fin",
        "French": "fra",
        "Hebrew": "heb",
        "Hindi": "hin",
        "Hungarian": "hun",
        "Indonesian": "ind",
        "Italian": "ita",
        "Dutch": "nld",
        "Polish": "pol",
        "Portuguese": "por",
        "Russian": "rus",
        "Spanish": "spa",
        "Slovak": "slk",
        "Swahili": "swh",
        "Tagalog": "tgl",
        "Turkish": "tur",
        "Urdu": "urd",
        "Vietnamese": "vie",
    }

    name = "mutox"
    description = "Toxicity detection benchmark ([Costa-jussà et al, 2024](https://arxiv.org/abs/2401.05060))."
    tags = ["audio", "classification", "toxicity "]

    @staticmethod
    def track_bad_audio_file(bad_audio_file: str, output_path: str) -> None:
        """
        Many of the links do not exist or point to broken so we keep track of them
        and skip them in the future runs to significantly speed up gathering the instances.
        """
        with open(output_path, "a") as f:
            f.write(bad_audio_file + "\n")

    def __init__(self, language: str) -> None:
        super().__init__()
        self._language_code: str = self.LANGAUGE_CODES[language]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the annotations
        annotations_path: str = os.path.join(output_path, "mutox.tsv")
        ensure_file_downloaded(self.ANNOTATIONS_URL, annotations_path)

        # Read bad audio files
        bad_audio_files: set[str] = set()
        bad_audio_files_path: str = os.path.join(output_path, "bad_audio_files.txt")
        if os.path.exists(bad_audio_files_path):
            # Each line is the audio file name
            with open(bad_audio_files_path, "r") as f:
                for line in f:
                    bad_audio_files.add(line.strip())
            hlog(f"Found {len(bad_audio_files)} bad audio files.")

        # Where the audio files will be downloaded to
        audio_path: str = os.path.join(output_path, "audio")
        ensure_directory_exists(audio_path)

        instances: List[Instance] = []
        df = pd.read_csv(annotations_path, delimiter="\t")
        hlog(f"Found {len(df)} rows in the dataset")

        valid_count: int = 0
        total_count: int = 0
        for row in tqdm(df.itertuples()):
            # Only proces examples that are in devtest and the language we're interested in
            if row.partition != "devtest":
                continue

            if row.lang != self._language_code:
                continue

            total_count += 1

            # Discard known bad audio files
            audio_filename: str = f"{row.id}.mp3"
            with htrack_block(f"Processing audio file: {audio_filename}"):
                if audio_filename in bad_audio_files:
                    hlog(f"Skipping this example -- known bad audio file: {audio_filename}")
                    continue

                local_audio_path: str = os.path.join(audio_path, audio_filename)
                if not os.path.exists(local_audio_path):
                    # The provided URL has the complete audio, so we need to download it and clip it
                    # public_url_segment: a string formatted as url:start:end,
                    if not isinstance(row.public_url_segment, str):
                        # Sometimes URL is just a float causing an error. Skip those.
                        hlog(f"Skipping this example -- invalid URL: {row.public_url_segment}")
                        continue

                    parts = row.public_url_segment.split()
                    if len(parts) != 3:
                        hlog(f"Skipping this example -- invalid URL: {row.public_url_segment}")
                        continue

                    audio_url, start_ms_str, end_ms_str = parts
                    start_ms, end_ms = int(start_ms_str), int(end_ms_str)

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
                        hlog(
                            f"Skipping this example -- error downloading / processing audio file from {audio_url}: {e}"
                        )
                        self.track_bad_audio_file(audio_filename, bad_audio_files_path)
                        continue

                if is_invalid_audio_file(local_audio_path):
                    hlog(f"Skipping this example -- Invalid audio file even after downloading at {local_audio_path}")
                    self.track_bad_audio_file(audio_filename, bad_audio_files_path)
                    continue
                else:
                    valid_count += 1
                    hlog(f"Valid audio file found at {local_audio_path} (valid/total: {valid_count}/{total_count})")

            input = Input(
                multimedia_content=MultimediaObject(
                    media_objects=[
                        # Following what was prompted annotators as outlined in the paper
                        MediaObject(text="Does the utterance contain toxicity?", content_type="text/plain"),
                        MediaObject(location=local_audio_path, content_type="audio/mpeg"),
                    ]
                )
            )

            is_toxic: bool = row.label == 1
            references = [
                Reference(Output(text="Yes"), tags=[CORRECT_TAG] if is_toxic else []),
                Reference(Output(text="No"), tags=[CORRECT_TAG] if not is_toxic else []),
            ]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))

        assert len(instances) > 0, f"No instances found for language: {self._language_code}"
        return instances
