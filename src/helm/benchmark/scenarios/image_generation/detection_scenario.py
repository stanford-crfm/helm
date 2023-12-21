from typing import Dict, List
import json
import os
import csv

from helm.common.general import ensure_file_downloaded
from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT, Reference, Output, CORRECT_TAG


class DetectionScenario(Scenario):
    """
    This metrics measures whether generated images follows the specification of
    objects and their relations in the text prompts.

    The following three skills, as defined in DALL-EVAL being evaluated:
    1. "Object". Given a text prompt "a photo of OBJ", whether the generated image
    contains OBJ.
    2. "Count". Given a text prompt "a photo of COUNT OBJ", whether the generated image
    contains OBJ and whether its number matches COUNT.
    3. "Spatial". Given a text prompt "a photo of OBJ1 and OBJ2; OBJ1 is RELATION OBJ2",
    whether the generated image contains OBJ1 and OBJ2, and whether their spatial relation
    matches RELATION.

    We use a pre-trained ViTDet (ViT-B) as the detection backbone.

    Paper:
    [DALL-EVAL](https://arxiv.org/abs/2202.04053).
    [ViTDet](https://arxiv.org/abs/2203.16527).
    """

    DATASET_DOWNLOAD_URL: str = "https://drive.google.com/uc?export=download&id=1HwfBlZCbfO8Vwss4HEXcyyD5sVezpmPg"

    name = "detection"
    description = "A benchmark to measure the accuracy of objects and relations in generated images."
    tags = ["text-to-image"]

    def __init__(self, skill: str):
        super().__init__()
        assert skill in ["count", "spatial", "object"], f"Invalid skill: {skill}"
        self._selected_skill: str = skill

    def get_instances(self, output_path: str) -> List[Instance]:
        prompts_path: str = os.path.join(output_path, "prompts.csv")
        ensure_file_downloaded(source_url=self.DATASET_DOWNLOAD_URL, target_path=prompts_path)

        instances: List[Instance] = []

        with open(prompts_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for i, row in enumerate(csv_reader):
                if i == 0:
                    # Skip the header
                    continue

                skill: str = row[0]
                if skill != self._selected_skill:
                    continue

                prompt: str = row[1]
                obj1: str = row[2]
                if skill == "count":
                    count: int = int(row[4])
                if skill == "spatial":
                    obj2: str = row[3]
                    relation: str = row[5]

                references: Dict
                if skill == "object":
                    references = {"object": obj1}
                elif skill == "count":
                    references = {"count": count, "object": obj1}
                elif skill == "spatial":
                    references = {"objects": [obj1, obj2], "relation": relation}

                instance = Instance(
                    Input(text=prompt),
                    references=[Reference(output=Output(text=json.dumps(references)), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                    sub_split=skill,
                )
                instances.append(instance)

        return instances
