from typing import List
import os

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
from helm.common.images_utils import generate_hash


class MMStarScenario(Scenario):
    """
    MM-STAR is an elite vision-indispensable multi-modal benchmark comprising 1,500 challenge samples meticulously
    selected by humans. MMStar is designed to benchmark 6 core capabilities and 18 detailed axes, aiming to evaluate
    the multi-modal capacities of LVLMs with a carefully balanced and purified selection of samples. The samples
    are first roughly selected from current benchmarks with an automated pipeline, strict human review is then
    involved to ensure each selected sample exhibits visual dependency, minimal data leakage, and requires advanced
    multi-modal capabilities for the solution.

    Website: https://mmstar-benchmark.github.io/

    @article{chen2024we,
      title={Are We on the Right Way for Evaluating Large Vision-Language Models?},
      author={Chen, Lin and Li, Jinsong and Dong, Xiaoyi and Zhang, Pan and Zang, Yuhang and Chen, Zehui and Duan,
      Haodong and Wang, Jiaqi and Qiao, Yu and Lin, Dahua and others},
      journal={arXiv preprint arXiv:2403.20330},
      year={2024}
    }
    """

    HUGGINGFACE_DATASET_NAME: str = "Lin-Chen/MMStar"

    VALID_CATEGORIES: List[str] = [
        "coarse perception",
        "fine-grained perception",
        "instance reasoning",
        "logical reasoning",
        "math",
        "science technology",
    ]

    name = "mm_star"
    description = (
        "MM-STAR is an elite vision-indispensable multi-modal benchmark comprising 1,500 challenge samples "
        "meticulously selected by humans."
        "([Chen, 2024](https://arxiv.org/abs/2403.20330))."
    )
    tags = ["vision-language", "knowledge", "reasoning"]

    def __init__(self, category: str):
        super().__init__()

        category = category.replace("_", " ")
        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Valid categories are: {self.VALID_CATEGORIES}")
        if category == "science technology":
            category = "science & technology"

        self._category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []

        for row in tqdm(load_dataset(self.HUGGINGFACE_DATASET_NAME, split="val", cache_dir=output_path)):
            # Filter by category
            category: str = row["category"]
            if category != self._category:
                continue

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
                Instance(Input(multimedia_content=MultimediaObject(content)), references=references, split=VALID_SPLIT)
            )

        return instances
