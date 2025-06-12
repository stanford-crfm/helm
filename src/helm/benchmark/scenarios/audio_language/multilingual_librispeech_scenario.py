from typing import List
import os

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)
from tqdm import tqdm
from datasets import load_dataset
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.audio_utils import ensure_audio_file_exists_from_array


class MultilingualLibriSpeechScenario(Scenario):
    """Multilingual Librispeech

    The Multilingual LibriSpeech (Pratap et al, 2020) dataset is derived from read audiobooks
    from LibriVox and consists of 8 languages, including about 44.5K hours of English and a total
    of about 6K hours for other 7 languages. The task is to recognize the textual content from the
    audio sample.

    Paper: https://arxiv.org/abs/2012.03411
    Code: https://www.openslr.org/

    Citation:
    @article{Pratap2020MLSAL,
        title={MLS: A Large-Scale Multilingual Dataset for Speech Research},
        author={Vineel Pratap and Qiantong Xu and Anuroop Sriram and Gabriel Synnaeve and Ronan Collobert},
        journal={ArXiv},
        year={2020},
        volume={abs/2012.03411}
        }
    """

    HF_DATASET_NAME = "facebook/multilingual_librispeech"
    LANGUAGE_LIST: List[str] = ["dutch", "german", "french", "spanish", "italian", "portuguese", "polish"]

    name = "multilingual_librispeech"
    description = (
        "Speech recognition in 7 different languages ([Pratap et al, 2022](https://arxiv.org/abs/2012.03411))."
    )
    tags: List[str] = ["audio", "multilinguality", "recognition"]

    def __init__(self, language: str) -> None:
        super().__init__()

        language = language.lower()
        if language not in MultilingualLibriSpeechScenario.LANGUAGE_LIST:
            raise ValueError(f"Invalid language. Valid languages are: {MultilingualLibriSpeechScenario.LANGUAGE_LIST}")

        self._language: str = language

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        audio_save_dir = os.path.join(output_path, "audio_files")
        for idx, row in enumerate(
            tqdm(
                load_dataset(
                    MultilingualLibriSpeechScenario.HF_DATASET_NAME,
                    name=self._language,
                    cache_dir=output_path,
                    split=TEST_SPLIT,
                )
            )
        ):
            local_audio_path = os.path.join(audio_save_dir, str(idx) + "_" + row["original_path"].split("/")[-1])
            # download to the local path
            ensure_audio_file_exists_from_array(local_audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])
            answer = row["transcript"]
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/mpeg", location=local_audio_path)])
            )
            references = [Reference(Output(text=str(answer)), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
