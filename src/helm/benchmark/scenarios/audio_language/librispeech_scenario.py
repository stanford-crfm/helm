from typing import List
import os
import json

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
from helm.common.general import ensure_file_downloaded


class LibriSpeechScenario(Scenario):
    """LibriSpeech Corpus
    The LibriSpeech corpus (Vassil et al. 2015) is derived from audiobooks that are part of the LibriVox
    project, and contains 1000 hours of speech sampled at 16 kHz. The data has separately prepared language-model
    training data and pre-built language models. This corpus is one of the most widely-used ASR corpus, which
    has been extended to many applicaitons such as robust ASR and multilingual ASR tasks.

    Paper: https://ieeexplore.ieee.org/document/7178964
    Code: https://www.openslr.org/12

    Citation:
    @INPROCEEDINGS{7178964,
        author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
        booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        title={Librispeech: An ASR corpus based on public domain audio books},
        year={2015},
        doi={10.1109/ICASSP.2015.7178964}}
    """

    HF_DATASET_NAME = "openslr/librispeech_asr"
    HF_MAPPING_URL = (
        "https://huggingface.co/datasets/PahaII/SRB_instance_key_mapping/resolve/main/srb_instance_keys.json"
    )
    SRB_KEY = "srb_librispeech_noises_key2audio"
    SRB_SUBSET = "gnoise.1"
    MAPPING_KEY = "librispeech_id2line"

    name = "librispeech"
    description = (
        "Widely-used speech corpus for the speech recognition task "
        "([Vassil et al. 2015](https://ieeexplore.ieee.org/document/7178964))."
    )
    tags: List[str] = ["audio", "recognition"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        audio_save_dir = os.path.join(output_path, "audio_files")
        mapping_local_path = os.path.join(output_path, "srb_instance_keys.json")
        ensure_file_downloaded(source_url=LibriSpeechScenario.HF_MAPPING_URL, target_path=mapping_local_path)
        meta_data = load_dataset(
            LibriSpeechScenario.HF_DATASET_NAME,
            name="clean",
            cache_dir=output_path,
            split=TEST_SPLIT,
        )
        mapping_dict = json.load(open(mapping_local_path))
        srb_mapping_keys = mapping_dict[self.SRB_KEY][self.SRB_SUBSET]
        index2line_num = mapping_dict[self.MAPPING_KEY]
        for line_num in tqdm(list(srb_mapping_keys)):
            row = meta_data[int(index2line_num[line_num])]
            local_audio_name = f"{self.name}_{line_num}.mp3"
            local_audio_path = os.path.join(audio_save_dir, local_audio_name)
            ensure_audio_file_exists_from_array(local_audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])
            answer = row["text"].lower()
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/mp3", location=local_audio_path)])
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
