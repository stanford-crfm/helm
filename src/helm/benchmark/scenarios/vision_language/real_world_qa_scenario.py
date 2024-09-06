from typing import List
import os

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.images_utils import generate_hash


class RealWorldQAScenario(Scenario):
    """
    RealWorldQA is a benchmark designed for real-world understanding. The dataset consists of anonymized
    images taken from vehicles, in addition to other real-world images.

    Blog post: https://x.ai/blog/grok-1.5v
    Website: https://huggingface.co/datasets/xai-org/RealworldQA
    """

    HUGGINGFACE_DATASET_NAME: str = "xai-org/RealworldQA"

    name = "real_world_qa"
    description = (
        "A benchmark designed to to evaluate real-world spatial understanding capabilities of multimodal models "
        "([xAI, 2024](https://x.ai/blog/grok-1.5v))."
    )
    tags = ["vision-language", "knowledge", "reasoning"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for row in tqdm(load_dataset(self.HUGGINGFACE_DATASET_NAME, split=TEST_SPLIT, cache_dir=output_path)):
            # Save the image to disk
            image = row["image"]
            image_file_name: str = generate_hash(image) + ".jpg"
            local_image_path: str = os.path.join(output_path, image_file_name)
            if not os.path.exists(local_image_path):
                image.save(local_image_path)

            content: List[MediaObject] = [
                MediaObject(location=local_image_path, content_type="image/jpeg"),
                MediaObject(text=row["question"], content_type="text/plain"),
            ]
            references: List[Reference] = [Reference(output=Output(text=row["answer"]), tags=[CORRECT_TAG])]
            instances.append(
                Instance(Input(multimedia_content=MultimediaObject(content)), references=references, split=TEST_SPLIT)
            )

        return instances
