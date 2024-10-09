import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from helm.common.general import ensure_file_downloaded
from helm.common.media_object import MediaObject, MultimediaObject
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    VALID_SPLIT,
)


class MSCOCOCaptioningScenario(Scenario):
    """
    Microsoft COCO (MS-COCO) is a large-scale object detection, segmentation, and captioning dataset.
    It has 330K images, with over 200K of them labeled. We use the 2014 version of the dataset instead
    of the 2017 version because of the larger validation set. According to https://cocodataset.org/#download,
    the 2014 version has 83K images in the train split and 41K in the val split.

    Each image also has five captions. For example, image #335111 has the following five captions:
        1. a row of bikes on the sidewalk, 2 on the ground.
        2. a couple of bikes laying on their sides on a sidewalk.
        3. a person wearing a black coat with a hood stands on the street, near many bikes
        4. a woman standing in front of a row of bicycles in front of a bus stop with two bikes knocked over
        5. there are some bicycles laying on their sides

    Paper: https://arxiv.org/abs/1405.0312
    Website: https://cocodataset.org/#home
    """

    ANNOTATIONS_DOWNLOAD_URL: str = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    SPLIT_DOWNLOAD_URL_TEMPLATE: str = "http://images.cocodataset.org/zips/{split}2014.zip"
    COCO_SPLIT_TO_HELM_SPLIT: Dict[str, str] = {"train": TRAIN_SPLIT, "val": VALID_SPLIT}

    name = "mscoco"
    description = "Microsoft COCO: Common Objects in Context ([paper](https://arxiv.org/abs/1405.0312))."
    tags = ["text-to-image", "image-to-text"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the annotations which contains the image IDs, filenames and captions
        data_path: str = os.path.join(output_path, "data")
        ensure_file_downloaded(source_url=self.ANNOTATIONS_DOWNLOAD_URL, target_path=data_path, unpack=True)

        instances: List[Instance] = []
        for coco_split, helm_split in self.COCO_SPLIT_TO_HELM_SPLIT.items():
            # Download the images of the split
            split_url: str = self.SPLIT_DOWNLOAD_URL_TEMPLATE.format(split=coco_split)
            split_path: str = os.path.join(data_path, coco_split)
            ensure_file_downloaded(source_url=split_url, target_path=split_path, unpack=True)

            # Read the metadata for the split
            metadata_path: str = os.path.join(data_path, f"captions_{coco_split}2014.json")
            with open(metadata_path, "r") as f:
                metadata: Dict[str, Any] = json.load(f)

            # Get the path of each image
            image_id_to_path: Dict[int, str] = {
                image_metadata["id"]: os.path.join(split_path, image_metadata["file_name"])
                for image_metadata in metadata["images"]
            }

            # Gather the five captions for each image
            image_id_to_captions: Dict[int, List[str]] = defaultdict(list)
            for annotation in metadata["annotations"]:
                image_id_to_captions[annotation["image_id"]].append(annotation["caption"])

            # Create instances
            for image_id in image_id_to_path:
                image_path: str = image_id_to_path[image_id]
                captions: List[str] = image_id_to_captions[image_id]

                content: List[MediaObject] = [
                    MediaObject(location=image_path, content_type="image/jpeg"),
                ]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=[
                            Reference(Output(text=caption.rstrip()), tags=[CORRECT_TAG]) for caption in captions
                        ],
                        split=helm_split,
                    )
                )

        return instances
