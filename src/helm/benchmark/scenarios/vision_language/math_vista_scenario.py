import os
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.general import ensure_directory_exists


class MathVistaScenario(Scenario):
    """
    MathVista: Evaluating Math Reasoning in Visual Contexts

    To bridge this gap, we present MathVista, a benchmark designed to combine challenges from diverse
    mathematical and visual tasks. It consists of 6,141 examples, derived from 28 existing multimodal datasets
    involving mathematics and 3 newly created datasets (i.e., IQTest, FunctionQA, and PaperQA). Completing these
    tasks requires fine-grained, deep visual understanding and compositional reasoning, which all state-of-the-art
    foundation models find challenging.

        @inproceedings{lu2024mathvista,
          author    = {Lu, Pan and Bansal, Hritik and Xia, Tony and Liu, Jiacheng and Li, Chunyuan and Hajishirzi,
                       Hannaneh and Cheng, Hao and Chang, Kai-Wei and Galley, Michel and Gao, Jianfeng},
          title     = {MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts},
          booktitle={International Conference on Learning Representations (ICLR)},
          year      = {2024}
        }

    Paper: https://arxiv.org/abs/2310.02255
    Website: https://mathvista.github.io/
    """

    HUGGINGFACE_DATASET_NAME: str = "AI4Math/MathVista"

    # Only the testmini split has answers
    SPLIT: str = "testmini"

    # Supported difficulties
    GRADES: List[str] = ["elementary_school", "high_school", "college", "daily_life"]
    QUESTION_TYPES: List[str] = ["multi_choice", "free_form"]

    name = "math_vista"
    description = (
        "A benchmark designed to combine challenges from diverse mathematical and visual tasks. "
        "([Lu et al., 2024](https://arxiv.org/abs/2310.02255))."
    )
    tags = ["vision-language", "reasoning", "math"]

    def __init__(self, grade: str, question_type: str):
        super().__init__()
        assert grade in self.GRADES, f"Not supported: {grade}"
        self._grade: str = grade.replace("_", " ")

        assert question_type in self.QUESTION_TYPES, f"Invalid question type: {question_type}"
        self._question_type: str = question_type

    def get_instances(self, output_path: str) -> List[Instance]:
        ensure_directory_exists(os.path.join(output_path, "images"))
        instances: List[Instance] = []

        for row in tqdm(load_dataset(self.HUGGINGFACE_DATASET_NAME, split=self.SPLIT, cache_dir=output_path)):
            # Filter out the questions by type and grade (or difficulty)
            if row["question_type"] != self._question_type or row["metadata"]["grade"] != self._grade:
                continue

            pid: str = row["pid"]
            question: str = row["question"]
            answer: str = row["answer"]

            # Save the image locally
            assert row["image"] == f"images/{pid}.jpg", f"Invalid image path: {row['image']} for question {pid}"
            image_path: str = os.path.join(output_path, row["image"])

            if not os.path.exists(image_path):
                image = row["decoded_image"]
                if image.mode in ("RGBA", "P", "LA"):
                    image = image.convert("RGB")
                image.save(image_path)

            content: List[MediaObject] = [
                MediaObject(text=question, content_type="text/plain"),
                MediaObject(location=image_path, content_type="image/jpeg"),
            ]

            # Add the references
            references: List[Reference] = []
            if self._question_type == "multi_choice":
                options: List[str] = row["choices"]
                for option in options:
                    references.append(Reference(Output(text=option), tags=[CORRECT_TAG] if option == answer else []))
            else:
                references.append(Reference(Output(text=answer), tags=[CORRECT_TAG]))

                if row["unit"] is not None:
                    references.append(Reference(Output(text=f"{answer} {row['unit']}"), tags=[CORRECT_TAG]))

            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=references,
                    split=TEST_SPLIT,
                )
            )

        assert (
            len(instances) > 0
        ), f"No instances found for subject {self._grade} and question type {self._question_type}"
        return instances
