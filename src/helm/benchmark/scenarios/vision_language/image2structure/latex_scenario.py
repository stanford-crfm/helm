import os.path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists
from .utils_latex import latex_to_image


class LatexScenario(Scenario):
    PROMPT: str = "Prease provide the LaTex code used to generate this image. Only generate the code relevant to what you see. Your code will be surrounded by all the imports necessary as well as the begin and end document delimiters."  # noqa: E501
    HUGGINGFACE_DATASET_NAME: str = "stanford-crfm/i2s-latex"
    MAX_NUM_ASSETS: int = 10
    SUBJECTS: List[str] = ["equation", "figure", "table", "plot", "algorithm"]

    name = "image2latex"
    description = "Evaluate multimodel models on Latex generation to recreate a provided image"
    tags = ["vision-language"]

    helm_split_to_huggingface_split = {
        TRAIN_SPLIT: "train",
        VALID_SPLIT: "validation",
    }

    def __init__(self, subject: str, recompile_prompt: bool = True, split: str = VALID_SPLIT):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject
        self._recompile_prompt: bool = recompile_prompt
        self._split: str = split

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "data/images", self._subject)
        assets_path: str = os.path.join(output_path, "data/assets")
        ensure_directory_exists(images_path)

        instances: List[Instance] = []

        # Process the desired set of instances
        for row in tqdm(
            load_dataset(
                self.HUGGINGFACE_DATASET_NAME,
                self._subject,
                split=self.helm_split_to_huggingface_split[self._split],
                cache_dir=output_path,
            )
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
                if not self._recompile_prompt:
                    row["output"].save(image_path)
                else:
                    latex_code: str = row["tex_code"]
                    image, _ = latex_to_image(latex_code, assets_path=assets_path, crop=True)
                    image.save(image_path)

            # Create the multimedia content
            prompt: str = self.PROMPT
            if len(asset_names) > 0:
                list_assets_str: str = "\n -".join(asset_names)
                prompt += f"\n\nThe following assets can be used:{list_assets_str}"
            image_object = MediaObject(location=image_path, content_type="image/png")
            content: List[MediaObject] = [
                MediaObject(text=prompt, content_type="text/plain"),
                image_object,
            ]

            # Create the reference
            reference: Reference = Reference(
                output=Output(
                    text=row["tex_code"],
                    multimedia_content=MultimediaObject(
                        [
                            image_object,
                            # TODO: Change this once we have a better way to pass the assets_path
                            # to the evaluation
                            MediaObject(text=f"assets_path={assets_path}", content_type="text/plain"),
                        ]
                    ),
                ),
                tags=[CORRECT_TAG],  # TODO: Add assets
            )

            # Finalize the Instance
            instances.append(
                Instance(
                    input=Input(multimedia_content=MultimediaObject(content)), references=[reference], split=self._split
                )
            )

        assert len(instances) > 0, f"No instances found for subject {self._subject}"
        return instances
