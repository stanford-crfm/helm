from typing import List
import os

from datasets import load_dataset
from tqdm import tqdm

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    VALID_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)
from helm.common.media_object import MediaObject, MultimediaObject
from helm.common.images_utils import generate_hash


class BlinkScenario(Scenario):
    """
    BLINK is a benchmark containing 14 visual perception tasks that can be solved by humans “within a blink”,
    but pose significant challenges for VLMs.

    Website: https://zeyofu.github.io/blink/

    @article{fu2024blink,
        title={BLINK: Multimodal Large Language Models Can See but Not Perceive},
        author={Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth,
        Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
        journal={arXiv preprint arXiv:2404.12390},
        year={2024}
    }
    """

    HUGGINGFACE_DATASET_NAME: str = "BLINK-Benchmark/BLINK"

    VALID_CATEGORIES: List[str] = [
        "Art_Style",
        "Counting",
        "Forensic_Detection",
        "Functional_Correspondence",
        "IQ_Test",
        "Jigsaw",
        "Multi-view_Reasoning",
        "Object_Localization",
        "Relative_Depth",
        "Relative_Reflectance",
        "Semantic_Correspondence",
        "Spatial_Relation",
        "Visual_Correspondence",
        "Visual_Similarity",
    ]

    name = "blink"
    description = (
        "BLINK is a benchmark containing 14 visual perception tasks that can be solved by humans within a blink, "
        "but pose significant challenges for VLMs. ([Fu, 2024](https://arxiv.org/abs/2404.12390))."
    )
    tags = ["vision-language", "knowledge", "reasoning"]

    def __init__(self, category: str):
        super().__init__()

        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Valid categories are: {self.VALID_CATEGORIES}")
        self._category: str = category

    def get_instances(self, output_path: str) -> List[Instance]:
        def save_image(image) -> str:
            image_file_name: str = generate_hash(image) + ".jpg"
            local_image_path: str = os.path.join(output_path, image_file_name)
            if not os.path.exists(local_image_path):
                image.save(local_image_path)
            return local_image_path

        def get_image_header(image_index: int) -> str:
            if image_index == 1:
                return "First image:"
            elif image_index == 2:
                return "Second image:"
            elif image_index == 3:
                return "Third image:"
            elif image_index == 4:
                return "Fourth image:"
            else:
                raise ValueError(f"Invalid image index: {image_index}")

        instances: List[Instance] = []
        for row in tqdm(
            load_dataset(self.HUGGINGFACE_DATASET_NAME, self._category, split="val", cache_dir=output_path)
        ):
            # Save the image(s) to disk
            has_multiple_images: bool = row["image_2"] is not None
            content: List[MediaObject] = []

            if has_multiple_images:
                # An example can have up to 4 images
                for i in range(1, 5):
                    image_i = row[f"image_{i}"]
                    if image_i is None:
                        break

                    # Before each image, include a header text that indicates which number image it is.
                    # Some prompts refer to specific image numbers within the question, e.g.,
                    # "Given three similar but different images, take the first image as reference.
                    # Can you tell which one of the latter two images is most similar to the first one?
                    # Select from the following choices. (A) the second image (B) the third image"
                    image_path: str = save_image(image_i)
                    content.extend(
                        [
                            MediaObject(text=get_image_header(i), content_type="text/plain"),
                            MediaObject(location=image_path, content_type="image/jpeg"),
                        ]
                    )
            else:
                image1 = row["image_1"]
                image1_path: str = save_image(image1)
                content.append(MediaObject(location=image1_path, content_type="image/jpeg"))

            # Add the prompt that has both the question and the answer choices
            prompt: str = row["prompt"]
            # Replace (A), (B), (C), (D) with \nA. \nB. \nC. \nD. since we are just expecting the letter answer
            prompt = prompt.replace("(A)", "\nA.").replace("(B)", "\nB.").replace("(C)", "\nC.").replace("(D)", "\nD.")
            content.append(MediaObject(text=prompt, content_type="text/plain"))

            # The answer has the correct letter choices surrounded by parentheses
            paren_letter_answer: str = row["answer"]
            assert (
                paren_letter_answer[0] == "(" and paren_letter_answer[-1] == ")"
            ), f"Unexpected answer format: {paren_letter_answer}"
            letter_answer: str = paren_letter_answer[1]
            references: List[Reference] = [
                Reference(output=Output(text=letter_answer), tags=[CORRECT_TAG]),
            ]
            instances.append(
                Instance(Input(multimedia_content=MultimediaObject(content)), references=references, split=VALID_SPLIT)
            )

        return instances
