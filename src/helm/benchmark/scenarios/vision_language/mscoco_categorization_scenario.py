import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Set

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


class MSCOCOCategorizationScenario(Scenario):
    """
    Microsoft COCO (MS-COCO) is a large-scale object detection, segmentation, and captioning dataset.
    It has 330K images, with over 200K of them labeled. We use the 2017 version of the dataset
    for the categorization task.

    Paper: https://arxiv.org/abs/1405.0312
    Website: https://cocodataset.org/#home
    """

    ANNOTATIONS_DOWNLOAD_URL: str = "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip"
    SPLIT_DOWNLOAD_URL_TEMPLATE: str = "http://images.cocodataset.org/zips/{split}2017.zip"
    COCO_SPLIT_TO_HELM_SPLIT: Dict[str, str] = {"train": TRAIN_SPLIT, "val": VALID_SPLIT}

    name = "mscoco"
    description = "Microsoft COCO: Common Objects in Context ([paper](https://arxiv.org/abs/1405.0312))."
    tags = ["text-to-image", "image-to-text"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the annotations which contains the image IDs, filenames and captions
        data_path: str = os.path.join(output_path, "data_2017")
        ensure_file_downloaded(source_url=self.ANNOTATIONS_DOWNLOAD_URL, target_path=data_path, unpack=True)

        super_categories_to_categories: Dict[str, List[str]] = defaultdict(list)
        category_id_to_category: Dict[int, str] = {}
        category_id_to_super_category: Dict[int, str] = {}

        instances: List[Instance] = []
        for coco_split, helm_split in self.COCO_SPLIT_TO_HELM_SPLIT.items():
            # Download the images of the split
            split_url: str = self.SPLIT_DOWNLOAD_URL_TEMPLATE.format(split=coco_split)
            split_path: str = os.path.join(data_path, coco_split)
            ensure_file_downloaded(source_url=split_url, target_path=split_path, unpack=True)

            # Read the metadata for the split
            metadata_path: str = os.path.join(data_path, f"stuff_{coco_split}2017.json")
            with open(metadata_path, "r") as f:
                metadata: Dict[str, Any] = json.load(f)

            for category_metadata in metadata["categories"]:
                # Each metadata looks like this {'supercategory': 'textile', 'id': 92, 'name': 'banner'}
                category_id: int = category_metadata["id"]
                category: str = category_metadata["name"]
                super_category: str = category_metadata["supercategory"]
                super_categories_to_categories[super_category].append(category)
                category_id_to_category[category_id] = category
                category_id_to_super_category[category_id] = super_category

            # Get the path of each image
            image_id_to_path: Dict[int, str] = {
                image_metadata["id"]: os.path.join(split_path, image_metadata["file_name"])
                for image_metadata in metadata["images"]
            }

            # Gather the five captions for each image
            image_id_to_category_ids: Dict[int, List[int]] = defaultdict(list)
            for annotation in metadata["annotations"]:
                image_id_to_category_ids[annotation["image_id"]].append(annotation["category_id"])

            # Create instances
            for image_id in image_id_to_path:
                image_path: str = image_id_to_path[image_id]
                assert os.path.exists(image_path), f"Image path {image_path} does not exist"
                category_ids: List[int] = image_id_to_category_ids[image_id]

                content: List[MediaObject] = [
                    MediaObject(location=image_path, content_type="image/jpeg"),
                ]
                references: List[Reference] = []
                correct_super_categories: Set[str] = set(
                    category_id_to_super_category[category_id] for category_id in category_ids
                )
                # for category_id in category_ids:
                #     category = category_id_to_category[category_id]
                #     super_category = category_id_to_super_category[category_id]
                #     references.extend(
                #         [
                #             Reference(Output(text=category), tags=[CORRECT_TAG]),
                #             Reference(Output(text=super_category), tags=[CORRECT_TAG]),
                #         ]
                #     )
                for super_category in super_categories_to_categories:
                    references.append(
                        Reference(
                            Output(text=super_category),
                            tags=[CORRECT_TAG] if super_category in correct_super_categories else [],
                        )
                    )

                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=references,
                        split=helm_split,
                    )
                )

        return instances
