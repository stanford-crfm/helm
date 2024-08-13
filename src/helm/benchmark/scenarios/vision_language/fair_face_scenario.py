from typing import List
import os

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    VALID_SPLIT,
    TRAIN_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.images_utils import generate_hash


class FairFaceScenario(Scenario):
    """
    Identify the race, gender or age of a photo of a person.

    @misc{kärkkäinen2019fairfacefaceattributedataset,
      title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age},
      author={Kimmo Kärkkäinen and Jungseock Joo},
      year={2019},
      eprint={1908.04913},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1908.04913},
    }

    Paper: https://arxiv.org/abs/1908.04913
    Website: https://huggingface.co/datasets/HuggingFaceM4/FairFace
    """

    HUGGINGFACE_DATASET_NAME: str = "HuggingFaceM4/FairFace"

    RACE_ATTRIBUTE: str = "race"
    GENDER_ATTRIBUTE: str = "gender"
    AGE_ATTRIBUTE: str = "age"

    QUESTION_TEMPLATE: str = "Identify the {attribute} of the person in the image."

    RACE_SUBGROUPS: List[str] = [
        "East Asian",
        "Indian",
        "Black",
        "White",
        "Middle Eastern",
        "Latino Hispanic",
        "Southeast Asian",
    ]
    GENDER_SUBGROUPS: List[str] = ["Male", "Female"]
    AGE_SUBGROUPS: List[str] = [
        "0-2 years",
        "3-9 years",
        "10-19 years",
        "20-29 years",
        "30-39 years",
        "40-49 years",
        "50-59 years",
        "60-69 years",
        "Over 70 years",
    ]

    name = "fair_face"
    description = (
        "Identify the race, gender or age of a photo of a person "
        "([Karkkainen et al., 2019](https://arxiv.org/abs/1908.04913))."
    )
    tags = ["vision-language", "fairness"]

    def __init__(self, attribute: str, subgroup: str) -> None:
        super().__init__()

        subgroups: List[str]
        if attribute == self.RACE_ATTRIBUTE:
            subgroups = self.RACE_SUBGROUPS
        elif attribute == self.GENDER_ATTRIBUTE:
            subgroups = self.GENDER_SUBGROUPS
        elif attribute == self.AGE_ATTRIBUTE:
            subgroups = self.AGE_SUBGROUPS
        else:
            raise ValueError(f"Invalid attribute: {attribute}")

        # Validate the value passed in for the subgroup argument and set possible subgroup choices.
        # The subgroup passed in has a _ for spaces in the string.
        subgroup = subgroup.replace("_", " ")
        assert subgroup in subgroups, f"Invalid subgroup for {attribute} attribute: {subgroup}"
        self._subgroup_choices: List[str] = subgroups
        self._correct_subgroup_index: int = subgroups.index(subgroup)

        self._attribute: str = attribute  # For answer column
        self._question: str = self.QUESTION_TEMPLATE.format(attribute=attribute)  # What text to prompt the model?

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for split in [TRAIN_SPLIT, VALID_SPLIT]:
            for row in tqdm(
                load_dataset(
                    self.HUGGINGFACE_DATASET_NAME,
                    "1.25",
                    split="validation" if split == VALID_SPLIT else split,
                    cache_dir=output_path,
                )
            ):
                # Filter out rows that do not match the subgroup
                if row[self._attribute] != self._correct_subgroup_index:
                    continue

                # Save the image to disk
                image = row["image"]
                image_file_name: str = generate_hash(image) + ".jpg"
                local_image_path: str = os.path.join(output_path, image_file_name)
                if not os.path.exists(local_image_path):
                    image.save(local_image_path)

                content: List[MediaObject] = [
                    MediaObject(location=local_image_path, content_type="image/jpeg"),
                    MediaObject(text=self._question, content_type="text/plain"),
                ]
                references: List[Reference] = [
                    Reference(
                        output=Output(text=subgroup),
                        tags=[CORRECT_TAG] if i == self._correct_subgroup_index else [],
                    )
                    for i, subgroup in enumerate(self._subgroup_choices)
                ]
                instances.append(
                    Instance(Input(multimedia_content=MultimediaObject(content)), references=references, split=split)
                )

        return instances
