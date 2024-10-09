import json
import os
from typing import Dict, List, Set

from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, CORRECT_TAG, VALID_SPLIT


class PaintSkillsScenario(Scenario):
    """
    PaintSkills is a compositional diagnostic dataset an evaluation toolkit that measures three
    fundamental visual reasoning capabilities:

    - object recognition => object
    - object counting => count
    - spatial relation understanding => spatial

    Paper: https://arxiv.org/abs/2202.04053
    Website: https://github.com/j-min/DallEval/tree/main/paintskills
    """

    METADATA_DOWNLOAD_URL: str = "https://drive.google.com/uc?export=download&id=12jsHDzEcBr-Et3FhLq-HckI5cmLB_rxC"
    SKILL_TO_DOWNLOAD_URL: Dict[str, str] = {
        "object": "https://drive.google.com/uc?export=download&id=1lpvSpBNfEg5EJt16prumXiuEO99byjzw&confirm=t",
        "count": "https://drive.google.com/uc?export=download&id=1koA-5xiZbAUDh65jpYaylG3IOA-mZTH2&confirm=t",
        "spatial": "https://drive.google.com/uc?export=download&id=1g-L0dVQjBTWp1uRwJLYXIj2xYIlQ2knu&confirm=t",
    }

    name = "paint_skills"
    description = (
        "A compositional diagnostic dataset an evaluation toolkit that measures visual reasoning skills "
        "([paper](https://arxiv.org/abs/2202.04053))."
    )
    tags = ["text-to-image", "image-to-text"]

    def __init__(self, skill: str):
        super().__init__()
        assert skill in self.SKILL_TO_DOWNLOAD_URL, f"Invalid skill: {skill}"
        self.skill: str = skill

    def get_instances(self, output_path: str) -> List[Instance]:
        skills_data_path: str = os.path.join(output_path, self.skill)
        ensure_file_downloaded(
            source_url=self.SKILL_TO_DOWNLOAD_URL[self.skill],
            target_path=skills_data_path,
            unpack=True,
            unpack_type="unzip",
        )

        images_path: str = os.path.join(skills_data_path, "images")
        with open(os.path.join(skills_data_path, "scenes", f"{self.skill}_val.json"), "r") as f:
            examples: Dict = json.load(f)

        instances: List[Instance] = []
        seen_captions: Set[str] = set()
        for example in examples["data"]:
            caption: str = example["text"]
            if caption in seen_captions:
                continue

            seen_captions.add(caption)
            image_path: str = os.path.join(images_path, f"image_{example['id']}.png")
            content: MultimediaObject = MultimediaObject([MediaObject(content_type="image/png", location=image_path)])
            instance = Instance(
                Input(text=caption),
                references=[Reference(Output(multimedia_content=content), tags=[CORRECT_TAG])],
                split=VALID_SPLIT,
            )
            instances.append(instance)

        return instances
