from typing import List, Set
import os

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    TRAIN_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.images_utils import generate_hash


class ExamsVScenario(Scenario):
    """
    EXAMS-V: A Multi-Discipline Multilingual Multimodal Exam Benchmark for Evaluating Vision Language Models

    A challenging multi-discipline multimodal multilingual exam benchmark for evaluating vision language models.
    It consists of 20,932 multiple-choice questions across 20 school disciplines covering natural science,
    social science, and other miscellaneous studies, e.g.,religion, fine arts, business, etc.

    Paper: https://arxiv.org/abs/2403.10378
    Website: https://huggingface.co/datasets/Rocktim/EXAMS-V
    """

    HUGGINGFACE_DATASET_NAME: str = "Rocktim/EXAMS-V"

    VALID_LANGUAGES: Set[str] = {
        "Chinese",
        "Croation",
        "Italian",
        "Hungarian",
        "Arabic",
        "Serbian",
        "Bulgarian",
        "English",
        "German",
        "French",
        "Spanish",
        "Polish",
    }
    VALID_SUBJECT_GROUP: Set[str] = {
        "Natural Science",
        "Social Sciences",
        "Other",
    }
    VALID_TYPES: Set[str] = {"text", "image_text"}

    name = "exams_v"
    description = (
        "Multimodal and Multilingual benchmark to evaluate vision-language models across 20 school disciplines "
        "covering natural science, social science, and other miscellaneous studies "
        "([Das et al., 2024]( https://arxiv.org/abs/2403.10378))."
    )
    tags = ["vision-language", "knowledge", "reasoning", "multilingual"]

    def __init__(self, language: str, subject_grouped: str, type: str) -> None:
        super().__init__()

        subject_grouped = subject_grouped.replace("_", " ")
        assert subject_grouped in self.VALID_SUBJECT_GROUP, f"Invalid subject_grouped: {subject_grouped}"
        assert type in self.VALID_TYPES, f"Invalid type: {type}"
        assert language in self.VALID_LANGUAGES, f"Invalid language: {language}"

        self._language: str = language
        self._subject_grouped: str = subject_grouped
        self._type: str = type

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []

        for split in [TRAIN_SPLIT, TEST_SPLIT]:
            for row in tqdm(load_dataset(self.HUGGINGFACE_DATASET_NAME, split=split, cache_dir=output_path)):
                language: str = row["language"]
                subject_grouped: str = row["subject_grouped"]
                type: str = row["type"]

                # Exclude examples that do not match the specified language, subject, and type
                if language != self._language or subject_grouped != self._subject_grouped or type != self._type:
                    continue

                # Save the image to disk
                image = row["image"]
                image_file_name: str = generate_hash(image) + ".jpg"
                local_image_path: str = os.path.join(output_path, image_file_name)
                if not os.path.exists(local_image_path):
                    image.convert("RGB").save(local_image_path)

                content: List[MediaObject] = [
                    MediaObject(location=local_image_path, content_type="image/jpeg"),
                ]
                references: List[Reference] = [Reference(output=Output(text=row["answer_key"]), tags=[CORRECT_TAG])]
                instances.append(
                    Instance(Input(multimedia_content=MultimediaObject(content)), references=references, split=split)
                )

        return instances
