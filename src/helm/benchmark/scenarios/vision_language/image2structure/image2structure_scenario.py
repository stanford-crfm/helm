import os.path
from typing import List, Optional
from abc import abstractmethod

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    VALID_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists
from helm.common.hierarchical_logger import hlog


class Image2StructureScenario(Scenario):
    BASE_PROMPT: str
    HUGGINGFACE_DATASET_NAME: str
    SUBSETS: List[str]

    name: str
    description: str
    tags = ["vision-language"]

    helm_split_to_huggingface_split = {
        TEST_SPLIT: "test",
        VALID_SPLIT: "validation",
    }

    def __init__(self, subset: str, recompile_prompt: bool = True, split: str = VALID_SPLIT):
        super().__init__()
        assert subset in self.SUBSETS, f"Invalid subset: {subset}"
        self._subset: str = subset
        self._recompile_prompt: bool = recompile_prompt
        self._split: str = split
        self._output_path: Optional[str] = None

    @abstractmethod
    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> None:
        pass

    def get_instances(self, output_path: str) -> List[Instance]:
        self._output_path = output_path
        images_path: str = os.path.join(output_path, "data/images", self._subset)
        assets_path: str = os.path.join(output_path, "data/assets", self._subset)
        ensure_directory_exists(images_path)

        instances: List[Instance] = []

        # Process the desired set of instances
        for row in tqdm(
            load_dataset(
                self.HUGGINGFACE_DATASET_NAME,
                self._subset,
                split=self.helm_split_to_huggingface_split[self._split],
                cache_dir=output_path,
            )
        ):
            question_id: str = row["num_id"]
            if row["category"][1:-1] != self._subset:
                hlog(
                    f"Skipping instance {question_id} as it belong in category"
                    f" {row['category']} and not {self._subset}"
                )
                continue

            # Save the image locally
            image_path: str = os.path.join(images_path, f"{question_id}.png")
            if not os.path.exists(image_path):
                if not self._recompile_prompt:
                    row["image"].save(image_path)
                else:
                    structure: str = row["structure"]
                    self.compile_and_save(structure, assets_path, image_path)

            # Create the multimedia content
            prompt: str = self.BASE_PROMPT
            image_object = MediaObject(location=image_path, content_type="image/png")
            content: List[MediaObject] = [
                MediaObject(text=prompt, content_type="text/plain"),
                image_object,
            ]

            # Create the reference
            reference: Reference = Reference(
                output=Output(
                    text=row["structure"] if len(row["structure"]) < 2000 else "",
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

        assert len(instances) > 0, f"No instances found for subject {self._subset}"
        return instances
