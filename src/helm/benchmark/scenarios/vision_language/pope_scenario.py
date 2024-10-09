from typing import List
import os

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from datasets import load_dataset
from tqdm import tqdm
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class POPEScenario(Scenario):
    """
    POPE dataset
    Despite the promising progress on Large Vision-Language Models (LVLMs), we find that LVLMs suffer from
    the hallucination problem, i.e. they tend to generate objects that are inconsistent with the target
    images in the descriptions. To investigate it, this work presents the first systematic study on object
    hallucination of LVLMs based on VQAv2 benchmark. We find that: objects that frequently occur in the
    visual instructions or co-occur with the image objects, are obviously prone to be hallucinated by LVLMs.
    In POPE, images from VQAv2 are matched with questions asking the appearance of certain objects in the
    image. We use the exact match metric for model evaluation on POPE.

    @inproceedings{li2023evaluating,
    title={Evaluating Object Hallucination in Large Vision-Language Models},
    author={Li, Yifan and Du, Yifan and Zhou, Kun and Wang, Jinpeng and Zhao, Wayne Xin and Wen, Ji-Rong},
    booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    pages={292--305},
    year={2023}
    }

    Paper: https://aclanthology.org/2023.emnlp-main.20/
    """

    POPE_HUGGINGFACE_DATASET_NAME: str = "lmms-lab/POPE"

    name = "pope"
    description = (
        "Open-ended questions about hallucination images "
        "([Li et al., 2023](https://aclanthology.org/2023.emnlp-main.20/))."
    )
    tags = ["vision-language", "visual question answering"]
    options: List[str] = ["Yes", "No"]

    def get_label_from_answer(self, answer: str):
        label: str
        if answer == "yes":
            label = "A"
        elif answer == "no":
            label = "B"
        else:
            raise NotImplementedError(f"Invalid answer: {answer}")
        return label

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)
        instances: List[Instance] = []
        for row in tqdm(
            load_dataset(
                self.POPE_HUGGINGFACE_DATASET_NAME,
                split=TEST_SPLIT,
                cache_dir=output_path,
            )
        ):
            image_source: str = row["image_source"]
            # Save the image locally
            image_path: str = os.path.join(images_path, f"{image_source}.jpg")
            if not os.path.exists(image_path):
                row["image"].save(image_path)

            question: str = row["question"]
            answer: str = row["answer"]
            references: List[Reference] = []

            answer = self.get_label_from_answer(answer)
            # The given correct answer is a letter, but we need an index
            correct_answer_index: int = ord(answer) - ord("A")
            # The options are originally appended to the question

            for i, option in enumerate(self.options):
                reference: Reference
                is_correct: bool = i == correct_answer_index
                reference = Reference(Output(text=option), tags=[CORRECT_TAG] if is_correct else [])
                references.append(reference)

            content = [
                MediaObject(location=image_path, content_type="image/jpeg"),
                MediaObject(text=question, content_type="text/plain"),
            ]
            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        return instances
