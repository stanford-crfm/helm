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


class AMIScenario(Scenario):
    """AMI Meeting Corpus
    The AMI Meeting Corpus (Carletta et al. 2005) is a multi-modal data set consisting of
    100 hours of meeting recordings. It is being created in the context of a project that
    is developing meeting browsing technology. The corpus is being recorded using a wide
    range of devices including close-talking and far-field microphones, individual and
    room-view video cameras, projection, a whiteboard, and individual pens, all of which
    produce output signals that are synchronized with each other.

    Paper: https://link.springer.com/chapter/10.1007/11677482_3
    Code: https://groups.inf.ed.ac.uk/ami/corpus/

    Citation:
    @inproceedings{Carletta2005TheAM,
        title={The AMI Meeting Corpus: A Pre-announcement},
        author={Jean Carletta and Simone Ashby and Sebastien Bourban and Mike Flynn and Ma{\"e}l Guillemot
        and Thomas Hain and Jaroslav Kadlec and Vasilis Karaiskos and Wessel Kraaij and Melissa Kronenthal
        and Guillaume Lathoud and Mike Lincoln and Agnes Lisowska Masson and Iain McCowan and Wilfried Post
        and Dennis Reidsma and Pierre D. Wellner},
        booktitle={Machine Learning for Multimodal Interaction},
        year={2005},
        url={https://api.semanticscholar.org/CorpusID:6118869}
        }
    """

    HF_DATASET_NAME = "edinburghcstr/ami"
    HF_MAPPING_URL = (
        "https://huggingface.co/datasets/PahaII/SRB_instance_key_mapping/resolve/main/srb_instance_keys.json"
    )
    name = "ami"
    description = (
        "Speech recognition of speech recorded with different devices "
        "([Carletta et al, 2005](https://link.springer.com/chapter/10.1007/11677482_3))."
    )
    SUBJECT_DICT = {
        "ihm": {"mapping_key": "ami_ihm_id2line", "srb_mapping": "nearfield"},
        "sdm": {"mapping_key": "ami_sdm_id2line", "srb_mapping": "farfield"},
    }
    tags: List[str] = ["audio", "recognition"]

    def __init__(self, subject: str) -> None:
        super().__init__()
        subject = subject.lower()
        if subject not in AMIScenario.SUBJECT_DICT.keys():
            raise ValueError(f"Invalid subject. Valid subjects are: {AMIScenario.SUBJECT_DICT.keys()}")

        self._subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        audio_save_dir = os.path.join(output_path, "audio_files")
        mapping_local_path = os.path.join(output_path, "srb_instance_keys.json")
        ensure_file_downloaded(source_url=AMIScenario.HF_MAPPING_URL, target_path=mapping_local_path)
        meta_data = load_dataset(
            AMIScenario.HF_DATASET_NAME,
            name=self._subject,
            cache_dir=output_path,
            split=TEST_SPLIT,
        )
        index_mappings = AMIScenario.SUBJECT_DICT[self._subject]["mapping_key"]
        srb_mappings = AMIScenario.SUBJECT_DICT[self._subject]["srb_mapping"]
        mapping_dict = json.load(open(mapping_local_path))
        srb_mapping_keys = mapping_dict["srb_aim_field_key2audio"][srb_mappings]
        index2line_num = mapping_dict[index_mappings]
        for line_num in tqdm(list(srb_mapping_keys)):
            row = meta_data[int(index2line_num[line_num])]
            local_audio_name = f"{self._subject}_{line_num}.wav"
            local_audio_path = os.path.join(audio_save_dir, local_audio_name)
            ensure_audio_file_exists_from_array(local_audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])
            answer = row["text"].lower()
            input = Input(
                multimedia_content=MultimediaObject([MediaObject(content_type="audio/wav", location=local_audio_path)])
            )
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG])]
            instances.append(Instance(input=input, references=references, split=TEST_SPLIT))
        return instances
