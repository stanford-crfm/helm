import os
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    VALID_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class Flickr30KScenario(Scenario):
    """
    An image caption corpus consisting of 158,915 crowdsourced captions describing 31,783 Flickr images.

    @article{young2014image,
      title={From image descriptions to visual denotations: New similarity metrics for semantic
             inference over event descriptions},
      author={Young, Peter and Lai, Alice and Hodosh, Micah and Hockenmaier, Julia},
      journal={Transactions of the Association for Computational Linguistics},
      volume={2},
      pages={67--78},
      year={2014},
      publisher={MIT Press}
    }

    Paper: https://shannon.cs.illinois.edu/DenotationGraph/TACLDenotationGraph.pdf
    Website: https://shannon.cs.illinois.edu/DenotationGraph/
    """

    HF_DATASET_NAME: str = "nlphuji/flickr30k"

    name = "flickr30k"
    description = (
        "An image caption corpus consisting of 158,915 crowd-sourced captions describing 31,783 Flickr "
        "images ([Young et al., 2014](https://shannon.cs.illinois.edu/DenotationGraph/TACLDenotationGraph.pdf))."
    )
    tags = ["vision-language"]

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)

        instances: List[Instance] = []
        for row in tqdm(load_dataset(self.HF_DATASET_NAME, cache_dir=output_path, split="test")):
            split: str = row["split"]
            helm_split: str = VALID_SPLIT if split == "val" else split

            image_filename: str = row["filename"]
            local_image_path: str = os.path.join(images_path, image_filename)
            image = row["image"]
            if not os.path.exists(local_image_path):
                image.save(local_image_path)

            content: List[MediaObject] = [
                MediaObject(location=local_image_path, content_type="image/jpeg"),
            ]
            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(Output(text=caption), tags=[CORRECT_TAG]) for caption in row["caption"]],
                    split=helm_split,
                )
            )

        return instances
