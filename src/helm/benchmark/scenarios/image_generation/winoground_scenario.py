from typing import List
import os

from datasets import load_dataset

from helm.common.general import get_file_name
from helm.common.images_utils import copy_image
from helm.common.media_object import MediaObject, MultimediaObject
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, CORRECT_TAG, TEST_SPLIT


class WinogroundScenario(Scenario):
    """
    Winoground is a novel task and dataset for evaluating the ability of vision and language models
    to conduct visio-linguistic compositional reasoning. Given two images and two captions, the
    goal is to match them correctly—but crucially, both captions contain a completely identical set
    of words/morphemes, only in a different order. The dataset was carefully hand-curated by
    expert annotators and is labeled with a rich set of fine-grained tags to assist in analyzing
    model performance.

    Users must agree to share their contact information before downloading the dataset from
    Hugging Face. Either agree to the terms and set HUGGING_FACE_ACCESS_TOKEN to an access token
    of a valid Hugging Face account or have the dataset pre-downloaded at the Hugging Face cache
    (default path: ~/.cache/huggingface/datasets).

    Paper: https://arxiv.org/abs/2204.03162
    Website: https://huggingface.co/datasets/facebook/winoground
    """

    name = "winoground"
    description = (
        "Winoground is a novel task and dataset for evaluating the ability of vision and language models "
        "to conduct visio-linguistic compositional reasoning "
        "([paper](https://arxiv.org/abs/2204.03162))."
    )
    tags = ["text-to-image", "image-to-text", "visual_reasoning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        auth_token: str = os.environ.get("HUGGING_FACE_ACCESS_TOKEN", "")

        instances: List[Instance] = []
        for row in load_dataset("facebook/winoground", split="test", use_auth_token=auth_token):
            # Use the first example of the pair for now (index 0)
            caption: str = row["caption_0"]
            image_path: str = row["image_0"].filename

            # Create a copy of the image in the benchmark output folder for metrics computation
            image_copy_path: str = os.path.join(output_path, get_file_name(image_path))
            if not os.path.exists(image_copy_path):
                copy_image(image_path, image_copy_path)
            content: MultimediaObject = MultimediaObject(
                [MediaObject(content_type="image/png", location=image_copy_path)]
            )

            instances.append(
                Instance(
                    input=Input(text=caption),
                    references=[Reference(Output(multimedia_content=content), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )
        return instances
