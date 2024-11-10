import datasets
import os
from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from helm.common.audio_utils import ensure_audio_file_exists_from_array
from helm.common.general import ensure_directory_exists
from helm.common.media_object import MediaObject, MultimediaObject


class IEMOCAPAudioScenario(Scenario):
    """IEMOCAP (Audio)

    This scenario is a emotion classification scenario based on the
    "Interactive emotional dyadic motion capture database" (IEMOCAP),
    collected by the Speech Analysis and Interpretation Laboratory (SAIL)
    at the University of Southern California (USC). Only the audio data
    from this dataset is used. The task is to classify the emotion of the
    speaker(s) in the audio sample as one of angry, happy, neutral or sad.

    Website: https://sail.usc.edu/iemocap/iemocap_release.htm
    Paper: https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf
    Dataset: https://huggingface.co/datasets/Zahra99/IEMOCAP_Audio/blob/main/README.md

    Citation:
    @article{article,
    author = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower Provost, Emily and Kim, Samuel and Chang, Jeannette and Lee, Sungbok and Narayanan, Shrikanth},
    year = {2008},
    month = {12},
    pages = {335-359},
    title = {IEMOCAP: Interactive emotional dyadic motion capture database},
    volume = {42},
    journal = {Language Resources and Evaluation},
    doi = {10.1007/s10579-008-9076-6}
    }
    """  # noqa: E501

    name = "iemocap_audio"
    description = "A classification scenario based on audio data from the Interactive emotional dyadic motion capture database (IEMOCAP) ([Busso et al, 2008](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf)). The task is to classify the emotion of the speaker(s) in the audio sample."  # noqa: E501
    tags = ["audio", "classification"]

    LABEL_NAMES = ["angry", "happy", "neutral", "sad"]
    SAMPLE_RATE = 16000

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        wav_dir = os.path.join(output_path, "wav")
        ensure_directory_exists(wav_dir)

        dataset = datasets.load_dataset(
            "Zahra99/IEMOCAP_Audio", revision="4f8539a397ecc0d7185bf941bc1bb7238abc3648", cache_dir=cache_dir
        )
        instances: List[Instance] = []
        for _, split in dataset.items():
            for row in split:
                wav_path = os.path.join(wav_dir, row["audio"]["path"])
                print(len(row["audio"]["array"]))
                print(list(row["audio"]["array"])[0:10])
                ensure_audio_file_exists_from_array(
                    wav_path, row["audio"]["array"], sample_rate=IEMOCAPAudioScenario.SAMPLE_RATE
                )
                input = Input(
                    multimedia_content=MultimediaObject(
                        media_objects=[MediaObject(location=wav_path, content_type="audio/wav")]
                    )
                )
                references = [
                    Reference(output=Output(text=IEMOCAPAudioScenario.LABEL_NAMES[row["label"]]), tags=[CORRECT_TAG])
                ]
                instance = Instance(input=input, references=references, split=TEST_SPLIT)
                instances.append(instance)
                print(row["audio"])
                break
        return instances
