import os
from typing import List

import pandas as pd
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.audio_utils import ensure_audio_file_exists_from_array, get_array_from_audio_file
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class MELDAudioScenario(Scenario):
    """Multimodal EmotionLines Dataset (MELD) Audio

    Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset.
    MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated
    in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -
    Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear.

    The task is to classify the emotion based on only the audio clip.

    Website: https://affective-meld.github.io/
    Paper: https://arxiv.org/abs/1810.02508
    Dataset: https://huggingface.co/datasets/DavidCombei/Wav2Vec_MELD_Audio

    Citation:
    S. Poria, D. Hazarika, N. Majumder, G. Naik, R. Mihalcea,
    E. Cambria. MELD: A Multimodal Multi-Party Dataset
    for Emotion Recognition in Conversation. (2018)

    Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W.
    EmotionLines: An Emotion Corpus of Multi-Party
    Conversations. arXiv preprint arXiv:1802.08379 (2018).
    """  # noqa: E501

    name = "meld_audio"
    description = "Classify emotions in audio clips from the television series Friends ([Poria et al, 2018](https://arxiv.org/abs/1810.02508))."  # noqa: E501
    tags = ["audio", "classification"]

    LABEL_NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    CSV_URL_PREFIX = (
        "https://raw.githubusercontent.com/declare-lab/MELD/2d2011b409d3ca2d7e94460cd007d434b1d0a102/data/MELD/"
    )
    SPLIT_NAME_TO_CSV_FILE_NAME = {
        VALID_SPLIT: "dev_sent_emo.csv",
        TRAIN_SPLIT: "train_sent_emo.csv",
        TEST_SPLIT: "test_sent_emo.csv",
    }
    SPLIT_NAME_TO_TGZ_FILE_NAME = {
        VALID_SPLIT: "audios_validation.tgz",
        TRAIN_SPLIT: "audios_train.tgz",
        TEST_SPLIT: "audios_test.tgz",
    }
    SAMPLE_RATE = 16000

    def get_instances(self, output_path: str) -> List[Instance]:
        csv_dir = os.path.join(output_path, "csv")
        ensure_directory_exists(csv_dir)

        instances: List[Instance] = []
        # Only download the test split.
        # We don't need the train split (because we use zero shot) or the validation split.
        split_name = TEST_SPLIT

        # Download the CSV to get the labels and IDs
        csv_file_name = MELDAudioScenario.SPLIT_NAME_TO_CSV_FILE_NAME[split_name]
        csv_file_path = os.path.join(csv_dir, csv_file_name)
        ensure_file_downloaded(MELDAudioScenario.CSV_URL_PREFIX + csv_file_name, csv_file_path)
        df = pd.read_csv(csv_file_path, header=0).rename(columns={"Sr No.": "serial_number"})

        # Download FLAC files
        flac_dir = os.path.join(output_path, f"flac_{split_name}")
        ensure_file_downloaded(
            source_url=f"https://huggingface.co/datasets/zrr1999/MELD_Text_Audio/resolve/main/archive/{MELDAudioScenario.SPLIT_NAME_TO_TGZ_FILE_NAME[split_name]}?download=true",  # noqa: E501
            target_path=flac_dir,
            unpack=True,
            unpack_type="untar",
        )

        wav_dir = os.path.join(output_path, f"wav_{split_name}")
        ensure_directory_exists(wav_dir)
        for row in tqdm(df.itertuples()):
            # Transcode FLAC to WAV
            wav_file_name = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.wav"
            wav_file_path = os.path.join(wav_dir, wav_file_name)
            if not os.path.isfile(wav_file_path):
                flac_file_name = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.flac"
                flac_file_path = os.path.join(flac_dir, flac_file_name)
                audio_array = get_array_from_audio_file(flac_file_path, MELDAudioScenario.SAMPLE_RATE)
                ensure_audio_file_exists_from_array(wav_file_path, audio_array, MELDAudioScenario.SAMPLE_RATE)
            input = Input(
                multimedia_content=MultimediaObject(
                    media_objects=[MediaObject(location=wav_file_path, content_type="audio/wav")]
                )
            )
            assert row.Emotion in MELDAudioScenario.LABEL_NAMES
            references = [Reference(output=Output(text=row.Emotion), tags=[CORRECT_TAG])]
            instance = Instance(
                id=str(f"awoo{row.serial_number}"), input=input, references=references, split=split_name
            )
            instances.append(instance)
        return instances
