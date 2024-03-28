from typing import List
import os

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from datasets import load_dataset
from tqdm import tqdm
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class POPEScenario(Scenario):
    """
    POPE dataset
    Despite the promising progress on Large Vision-Language Models (LVLMs), we find that LVLMs suffer from
    the hallucination problem, i.e. they tend to generate objects that are inconsistent with the target
    images in the descriptions. To investigate it, this work presents the first systematic study on object
    hallucination of LVLMs based on VQAv2 benchmark. We find that: objects that frequently occur in the
    visual instructions or co-occur with the image objects, are obviously prone to be hallucinated by LVLMs.
    In POPE, images from VQAv2 are matched with questions asking the appearance of certain objects in the
    image. We use the exact match metric for model evaluation on POPE.

    @inproceedings{li2023evaluating,
    title={Evaluating Object Hallucination in Large Vision-Language Models},
    author={Li, Yifan and Du, Yifan and Zhou, Kun and Wang, Jinpeng and Zhao, Wayne Xin and Wen, Ji-Rong},
    booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    pages={292--305},
    year={2023}
    }

    Paper: https://aclanthology.org/2023.emnlp-main.20/
    """

    POPE_HUGGINGFACE_DATASET_NAME: str = "lmms-lab/POPE"

    name = "pope"
    description = (
        "Open-ended questions about hallucination images ([paper](https://aclanthology.org/2023.emnlp-main.20/))."
    )
    tags = ["vision-language", "visual question answering"]

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)
        instances: List[Instance] = []
        for row in tqdm(
            load_dataset(
                self.POPE_HUGGINGFACE_DATASET_NAME,
                split=TEST_SPLIT,
                cache_dir=output_path,
            )
        ):
            image_source: str = row["image_source"]
            # Save the image locally
            image_path: str = os.path.join(output_path, f"{image_source}.jpg")
            if not os.path.exists(image_path):
                row["image"].save(image_path)

            content: List[MediaObject] = [
                MediaObject(location=image_path, content_type="image/jpeg"),
                MediaObject(text=row["question"], content_type="text/plain"),
            ]
            answer: str = row["answer"]
            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
