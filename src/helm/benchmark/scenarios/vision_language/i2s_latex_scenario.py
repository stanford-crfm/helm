import os.path
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


class LatexScenario(Scenario):
    PROMPT: str = "Prease provide the LaTex code used to generate this image. Only generate the code relevant to what you see. Your code will be surrounded by all the imports necessary as well as the begin and end document delimiters."  # noqa: E501
    HUGGINGFACE_DATASET_NAME: str = "JosselinSom/Latex-VLM"
    MAX_NUM_ASSETS: int = 10
    CATEGORIES: List[str] = ["equation", "figure", "table", "plot", "algorithm"]

    name = "i2s-latex"
    description = "Evaluate multimodel models on Latex generation to recreate a provided image"
    tags = ["vision-language"]

    def __init__(self, category: str):
        super().__init__()
        assert category in self.CATEGORIES, f"Invalid category: {category}"
        self._category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "i2s/latex/images", self._category)
        ensure_directory_exists(images_path)

        instances: List[Instance] = []

        # Process the validation set
        # There seems to be a dev set, but it's unavailable through load_dataset.
        # The test set doesn't have answers, since the MMMU competition/leaderboard uses the test set
        for row in tqdm(
            load_dataset(self.HUGGINGFACE_DATASET_NAME, "default", split="validation", cache_dir=output_path)
        ):
            question_id: str = row["id"]

            asset_names: List[str] = []
            for i in range(self.MAX_NUM_ASSETS):
                if row[f"asset_{i}"] is not None and len(row[f"asset_{i}"]) > 0:
                    asset_names.append(row[f"asset_{i}"])
                else:
                    # If asset_{i} is not specified, neither will be all the following
                    break

            # Save the image locally
            image_path: str = os.path.join(images_path, f"{question_id}.png")
            if not os.path.exists(image_path):
                row["output"].save(image_path)

            # Create the multimedia content
            content: List[MediaObject] = [
                MediaObject(text=self.PROMPT, content_type="text/plain"),
                MediaObject(location=image_path, content_type="image/png"),
            ]

            # Create the reference
            reference: Reference = Reference(
                output=Output(text=row["tex_code"], multimedia_content=None), tags=[CORRECT_TAG]  # TODO: Add assets
            )

            # Finalize the Instance
            instances.append(
                Instance(
                    input=Input(multimedia_content=MultimediaObject(content)), references=[reference], split=VALID_SPLIT
                )
            )

        assert len(instances) > 0, f"No instances found for category {self._category}"
        return instances
