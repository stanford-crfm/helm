import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.audio_utils import ensure_wav_file_exists_from_array
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class MELDScenario(Scenario):
    """Multimodal EmotionLines Dataset (MELD)

    Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset.
    MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated
    in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -
    Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear

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

    name = "meld"
    description = "Classify emotions in audio clips from the television series Friends ([Poria et al, 2018](https://arxiv.org/abs/1810.02508))."  # noqa: E501
    tags = ["audio", "classification"]

    LABEL_NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    SAMPLE_RATE = 16000

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        wav_dir = os.path.join(output_path, "wav")
        ensure_directory_exists(wav_dir)

        dataset = datasets.load_dataset(
            "DavidCombei/Wav2Vec_MELD_Audio", revision="3cc6387c53747a00ff8dd4cd307d04e7cea2dd87", cache_dir=cache_dir
        )
        instances: List[Instance] = []
        for split_name, split in dataset.items():
            if split_name == "train":
                # Skip train instances for now
                continue
            elif split_name == "validation":
                # Skip validation instances because we don't need them
                continue
            for row in split:
                wav_path = os.path.join(wav_dir, row["audio"]["path"])
                ensure_wav_file_exists_from_array(wav_path, row["audio"]["array"], sample_rate=MELDScenario.SAMPLE_RATE)
                input = Input(
                    multimedia_content=MultimediaObject(
                        media_objects=[MediaObject(location=wav_path, content_type="audio/wav")]
                    )
                )
                references = [Reference(output=Output(text=MELDScenario.LABEL_NAMES[row["label"]]), tags=[CORRECT_TAG])]
                instance = Instance(input=input, references=references, split=split_name)
                instances.append(instance)
        return instances
