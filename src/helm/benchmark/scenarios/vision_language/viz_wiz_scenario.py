from typing import Dict, List, Set
import json
import os

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
from helm.common.general import ensure_file_downloaded


class VizWizScenario(Scenario):
    """
    VizWiz is a real-world visual question answering dataset consisting of questions asked by people who are
    visually impaired. It originates from a natural visual question answering
    setting where blind people each took an image and recorded a spoken question about it,
    together with 10 crowdsourced answers per visual question.

    Version as of January 1, 2020:

    - 20,523 training image/question pairs
    - 205,230 training answer/answer confidence pairs
    - 4,319 validation image/question pairs
    - 43,190 validation answer/answer confidence pairs

    where answer confidences are one of {"yes", "maybe", "no"}.

    Answers are publicly shared for the train and validation splits and hidden for the test split.

    Paper: https://arxiv.org/abs/1802.08218
    Website: https://vizwiz.org/tasks-and-datasets/vqa
    """

    # Annotations are not available for the test set
    ANNOTATIONS_URL: str = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"
    SPLIT_TO_ANNOTATIONS_FILE: Dict[str, str] = {
        TRAIN_SPLIT: "train.json",
        VALID_SPLIT: "val.json",
    }

    SPLIT_TO_IMAGES: Dict[str, str] = {
        TRAIN_SPLIT: "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
        VALID_SPLIT: "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    }

    name = "viz_wiz"
    description = (
        "Real-world VQA dataset consisting of questions asked by "
        "people who are blind ([Gurari et al., 2018](https://arxiv.org/abs/1802.08218))."
    )
    tags = ["vision-language", "visual question answering"]

    def get_instances(self, output_path: str) -> List[Instance]:
        # Download the questions and annotations
        annotations_path: str = os.path.join(output_path, "annotations")
        ensure_file_downloaded(
            source_url=self.ANNOTATIONS_URL,
            target_path=annotations_path,
            unpack=True,
            unpack_type="unzip",
        )

        instances: List[Instance] = []
        for split in [TRAIN_SPLIT, VALID_SPLIT]:
            # Download the images for the split
            images_path: str = os.path.join(output_path, split)
            ensure_file_downloaded(
                source_url=self.SPLIT_TO_IMAGES[split],
                target_path=images_path,
                unpack=True,
                unpack_type="unzip",
            )

            annotations_split_path: str = os.path.join(annotations_path, self.SPLIT_TO_ANNOTATIONS_FILE[split])
            with open(annotations_split_path) as f:
                for image_annotation in json.load(f):
                    image_path: str = os.path.join(images_path, image_annotation["image"])
                    assert os.path.exists(image_path), f"Image {image_path} does not exist"

                    content: List[MediaObject] = [
                        MediaObject(location=image_path, content_type="image/jpeg"),
                        MediaObject(text=image_annotation["question"], content_type="text/plain"),
                    ]
                    deduped_answers: Set[str] = {
                        answer_json["answer"]
                        for answer_json in image_annotation["answers"]
                        if answer_json["answer_confidence"] == "yes"
                    }

                    instances.append(
                        Instance(
                            Input(multimedia_content=MultimediaObject(content)),
                            references=[
                                Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in deduped_answers
                            ],
                            split=split,
                        )
                    )

        return instances
