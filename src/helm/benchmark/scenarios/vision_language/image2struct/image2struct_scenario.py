import os.path
from typing import List, Optional, Dict, Any
from abc import abstractmethod

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    ASSET_NAME_TAG,
    ASSET_PATH_TAG,
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

PROCESSED: str = "processed"
DIFFICULTY_ALL = "all"
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"


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

    def __init__(
        self, subset: str, recompile_prompt: bool = True, split: str = VALID_SPLIT, difficulty: str = DIFFICULTY_ALL
    ):
        super().__init__()
        assert subset in self.SUBSETS, f"Invalid subset: {subset}"
        self._subset: str = subset
        self._recompile_prompt: bool = recompile_prompt
        self._split: str = split
        self._output_path: Optional[str] = None
        self._difficulty: str = difficulty

    def preprocess_row(self, row: Dict[str, Any], assets_path: str) -> Dict[str, Any]:
        # By default, there are no assets
        del row["assets"]
        row["assets_paths"] = []
        row["assets_names"] = []
        return row

    def build_prompt(self, row: Dict[str, Any]) -> str:
        return self.BASE_PROMPT

    @abstractmethod
    def compile_and_save(self, structure: str, assets_path: str, destination_path: str) -> str:
        """Compile the prompt, should save the image and return the text extracted from the image"""
        pass

    def finalize(self, row: Dict[str, Any]) -> None:
        """Perform cleanup operations after the instance has been generated."""
        pass

    def get_instances(self, output_path: str) -> List[Instance]:
        """Get the instances for the scenario. This compile_and_save method should be implemented by the subclass.
        Additionally, the subclass should implement the preprocess_row method if any preprocessing is needed.

        For each instance, the following steps are performed:
        1. Preprocess the row
        2. Save the image locally
            - 2.a. If we don't want to recompile the prompt, save the image directly
            - 2.b. If we want to recompile the prompt, compile the structure and save the image
        3. Create the prompt
        4. Create the multimedia content
        5. Create the reference
        6. Finalize the Instance

        Args:
            output_path (str): The path where the instances will be saved

        Returns:
            List[Instance]: The list of instances
        """
        self._output_path = output_path
        images_path: str = os.path.join(output_path, "data/images", self._subset)
        assets_path: str = os.path.join(output_path, "data/assets", self._subset)
        ensure_directory_exists(images_path)
        ensure_directory_exists(assets_path)

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
            question_uuid: str = str(row["uuid"]).replace('"', "")
            if row["category"][1:-1] != self._subset:
                hlog(
                    f"Skipping instance {question_uuid} as it belong in category"
                    f" {row['category']} and not {self._subset}"
                )
                continue

            # Filter by difficulty
            if self._difficulty != DIFFICULTY_ALL and row["difficulty"] != self._difficulty:
                continue

            # Step 1: Preprocess the row
            row = self.preprocess_row(row, assets_path)

            # Step 2: Save the image locally
            image_path: str = os.path.join(images_path, f"{question_uuid}.png")
            if not os.path.exists(image_path):
                if not self._recompile_prompt:  # 2.a
                    row["image"].save(image_path)
                else:  # 2.b
                    if "structure" not in row:
                        raise ValueError("Cannot recompile prompt without structure")
                    structure: str = row["structure"]
                    text: str = self.compile_and_save(structure, assets_path, image_path)
                    row["text"] = text

            # Step 3: Create the prompt
            prompt: str = self.build_prompt(row)

            # Step 4: Create the multimedia content
            image_object = MediaObject(location=image_path, content_type="image/png")
            content: List[MediaObject] = [
                MediaObject(text=prompt, content_type="text/plain"),
                image_object,
            ]

            # Step 5: Create the references
            # 5.a Create the reference containing the structure and the associated image.
            reference: Reference
            if "structure" in row:
                multimedia_object: MultimediaObject
                if os.path.exists(row["structure"]):
                    # 5.a.1 The structure is a path, therefore represent it as a multimedia object
                    # containing the files used to compile the structure (such as a repository
                    # containing the HTML, CSS, and JavaScript files used to generate a webpage)
                    multimedia_object = MultimediaObject(
                        [image_object, MediaObject(location=row["structure"], content_type="path/path")]
                    )
                elif row["structure"] == PROCESSED:
                    # 5.a.2 The structure has been processed and is no longer present in the row
                    # This can be the case if the structure is a base64 encoding of an archive that
                    # has been extracted to a temporary path and processed but the path is no longer
                    # existing (deleted after the processing is done)
                    multimedia_object = MultimediaObject([image_object])
                else:
                    # 5.a.3 The structure is not a path, therefore it is directly a valid string
                    # representing the structure (such as LaTeX code)
                    multimedia_object = MultimediaObject([image_object])
                reference = Reference(
                    output=Output(text=row["text"] if "text" in row else "", multimedia_content=multimedia_object),
                    tags=[CORRECT_TAG],
                )
            else:
                if "text" in row:
                    reference = Reference(
                        output=Output(text=row["text"], multimedia_content=MultimediaObject([image_object])),
                        tags=[CORRECT_TAG],
                    )
                else:
                    reference = Reference(
                        output=Output(multimedia_content=MultimediaObject([image_object])), tags=[CORRECT_TAG]
                    )
            references: List[Reference] = [reference]

            # 5.b Create the reference containing the assets
            if len(row["assets_paths"]) > 0:
                assets_paths_reference: Reference = Reference(
                    output=Output(
                        text=", ".join(
                            row["assets_paths"]
                        ),  # TODO: This is for debugging purposes (to show in the frontend)
                        multimedia_content=MultimediaObject(
                            [
                                MediaObject(location=asset, content_type=f"image/{asset.split('.')[-1].lower()}")
                                for asset in row["assets_paths"]
                            ]
                        ),
                    ),
                    tags=[ASSET_PATH_TAG],
                )
                references.append(assets_paths_reference)
                assets_names_reference: Reference = Reference(
                    output=Output(
                        text=", ".join(
                            row["assets_names"]
                        ),  # TODO: This is for debugging purposes (to show in the frontend)
                        multimedia_content=MultimediaObject(
                            [MediaObject(text=asset, content_type="text/plain") for asset in row["assets_names"]]
                        ),
                    ),
                    tags=[ASSET_NAME_TAG],
                )
                references.append(assets_names_reference)

            # Step 6: Finalize the Instance
            self.finalize(row)
            instance = Instance(
                input=Input(multimedia_content=MultimediaObject(content)), references=references, split=self._split
            )
            instances.append(instance)

        assert len(instances) > 0, f"No instances found for subject {self._subset}"
        return instances
