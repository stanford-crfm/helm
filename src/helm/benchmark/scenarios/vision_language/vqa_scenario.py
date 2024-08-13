from typing import Dict, List
import json
import os

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALID_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class VQAScenario(Scenario):
    """
    Visual Question Answering (VQA) is a dataset containing open-ended questions about images.
    These questions require an understanding of vision, language and commonsense knowledge to answer.

    We use the v2.0 version released in 2017 by Goyal et al.
    "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering".

    Balanced Real Images - 204,721 COCO images
    (all of current train/val/test)
    1,105,904 questions
    11,059,040 ground truth answers

    Paper:
      - v1: https://arxiv.org/abs/1505.00468
      - v2: https://arxiv.org/abs/1612.00837
    Website: https://visualqa.org
    """

    SPLIT_TO_QUESTIONS: Dict[str, str] = {
        TRAIN_SPLIT: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        VALID_SPLIT: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        TEST_SPLIT: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    }

    # Annotations are not available for the test set
    SPLIT_TO_ANNOTATIONS: Dict[str, str] = {
        TRAIN_SPLIT: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        VALID_SPLIT: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    }

    SPLIT_TO_IMAGES: Dict[str, str] = {
        TRAIN_SPLIT: "http://images.cocodataset.org/zips/train2014.zip",
        VALID_SPLIT: "http://images.cocodataset.org/zips/val2014.zip",
        TEST_SPLIT: "http://images.cocodataset.org/zips/test2015.zip",
    }

    name = "vqa"
    description = (
        "Open-ended questions about real-world images " "([Goyal et al., 2017](https://arxiv.org/abs/1612.00837))."
    )
    tags = ["vision-language", "visual question answering"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for split in [TRAIN_SPLIT, VALID_SPLIT]:
            # Download the questions and answers
            split_path: str = os.path.join(output_path, split)
            ensure_directory_exists(split_path)
            questions_path: str = os.path.join(split_path, "questions.json")
            ensure_file_downloaded(
                source_url=self.SPLIT_TO_QUESTIONS[split],
                target_path=questions_path,
                unpack=True,
                unpack_type="unzip",
            )
            with open(questions_path) as f:
                question_id_to_questions: Dict[str, Dict] = {
                    question_json["question_id"]: question_json for question_json in json.load(f)["questions"]
                }

            answers_path: str = os.path.join(split_path, "answers.json")
            ensure_file_downloaded(
                source_url=self.SPLIT_TO_ANNOTATIONS[split],
                target_path=answers_path,
                unpack=True,
                unpack_type="unzip",
            )
            with open(answers_path) as f:
                question_id_to_answers: Dict[str, Dict] = {
                    answers_json["question_id"]: answers_json for answers_json in json.load(f)["annotations"]
                }

            # Download the images
            images_path: str = os.path.join(split_path, "images")
            ensure_file_downloaded(
                source_url=self.SPLIT_TO_IMAGES[split],
                target_path=images_path,
                unpack=True,
                unpack_type="unzip",
            )

            for question_id, question_json in question_id_to_questions.items():
                answers_json = question_id_to_answers[question_id]

                assert (
                    question_json["image_id"] == answers_json["image_id"]
                ), "The image between question_json and answers_json does not match"
                image_id: str = str(answers_json["image_id"])
                image_id = "0" * (12 - len(image_id)) + image_id
                coco_split: str = "val" if split == VALID_SPLIT else split
                image_path: str = os.path.join(images_path, f"COCO_{coco_split}2014_{image_id}.jpg")
                content: List[MediaObject] = [
                    MediaObject(location=image_path, content_type="image/jpeg"),
                    MediaObject(text=question_json["question"], content_type="text/plain"),
                ]
                instances.append(
                    Instance(
                        Input(multimedia_content=MultimediaObject(content)),
                        references=[
                            Reference(Output(text=answer_json["answer"]), tags=[CORRECT_TAG])
                            for answer_json in answers_json["answers"]
                        ],
                        split=split,
                    )
                )

        return instances
