import os
from typing import List

from datasets import DatasetDict, load_dataset

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
)
from helm.common.general import ensure_directory_exists
from helm.common.media_object import MediaObject, MultimediaObject


class VQARadScenario(Scenario):
    """
    VQARad scenario: Processes a visual question answering dataset with radiology images.

    Each record in the dataset has:
    - image
    - question
    - answer

    The output is formatted as:
    "Answer: <answer>"
    """

    HUGGING_FACE_DATASET_PATH: str = "flaviagiammarino/vqa-rad"

    name = "vqa_rad"
    description = "Visual question answering with radiology images."
    tags = [
        "vision-language",
        "visual question answering",
        "reasoning",
        "medical",
        "radiology",
    ]

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset: DatasetDict = load_dataset(self.HUGGING_FACE_DATASET_PATH)

        splits = {TRAIN_SPLIT: "train", TEST_SPLIT: "test"}
        instances: List[Instance] = []
        # Iterate over the splits
        for (
            helm_split_name,
            dataset_split_name,
        ) in splits.items():
            split_path: str = os.path.join(output_path, dataset_split_name)
            ensure_directory_exists(split_path)

            split_data = dataset[dataset_split_name]

            for index, example in enumerate(split_data):
                question = example["question"]
                image = example["image"]
                answer = example["answer"]

                # Convert PIL image to MediaObject
                image_path = os.path.join(split_path, f"{index}.jpg")
                image.save(image_path)

                content = [
                    MediaObject(location=image_path, content_type="image/jpeg"),
                    MediaObject(text=question, content_type="text/plain"),
                ]

                # Format the final answer
                instances.append(
                    Instance(
                        input=Input(multimedia_content=MultimediaObject(content)),
                        references=[
                            Reference(
                                Output(text=answer),
                                tags=[CORRECT_TAG],
                            )
                        ],
                        split=helm_split_name,
                    )
                )

        return instances
