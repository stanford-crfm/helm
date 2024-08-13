import os.path
from typing import Dict, List

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
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class UnicornScenario(Scenario):
    """
    How Many Unicorns are in this Image? A Safety Evaluation Benchmark of Vision LLMs

    We shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation
    suite Unicorn, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD
    evaluation, we present two novel VQA datasets --- OODCV-VQA and Sketchy-VQA, each with one variant, designed
    to test model performance under challenging conditions. In the OOD scenario, questions are matched with
    boolean or numerical answers, and we use exact match metrics for evaluation. When comparing OOD Sketchy-VQA
    with its synthesized in-distribution counterpart, we found an average model output F1 drop of 8.9%,
    highlighting the challenging nature of the OOD scenario in the Unicorn benchmark.

    @article{tu2023unicorns,
    title={How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs},
    author={Tu, Haoqin and Cui, Chenhang and Wang, Zijun and Zhou, Yiyang and Zhao, Bingchen and Han,
    Junlin and Zhou, Wangchunshu and Yao, Huaxiu and Xie, Cihang},
    journal={arXiv preprint arXiv:2311.16101},
    year={2023}
    }

    Paper: https://arxiv.org/abs/2311.16101
    """

    UNICORN_HUGGINGFACE_DATASET_URL: str = "https://huggingface.co/datasets/PahaII/unicorn/resolve/main"

    IMAGE_URL: str = "https://huggingface.co/datasets/PahaII/unicorn/resolve/main/images/{image_path}?download=true"

    SUBJECTS: List[str] = ["OODCV-VQA", "OODCV-Counterfactual", "Sketchy-VQA", "Sketchy-Challenging"]

    IMG_TYPE: Dict[str, str] = {
        "OODCV-VQA": "jpeg",
        "OODCV-Counterfactual": "jpeg",
        "Sketchy-VQA": "png",
        "Sketchy-Challenging": "png",
    }

    name = "unicorn"
    description = (
        "Evaluate multimodal models on two out-of-distribution scenarios with four subjects "
        "([Tu et al., 2023](https://arxiv.org/abs/2311.16101))."
    )
    tags = ["vision-language"]

    def __init__(self, subject: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject
        self._image_type: str = self.IMG_TYPE[self._subject]

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)

        # There is only the test split in Unicorn benchmark
        instances: List[Instance] = []
        question_data_files = {TEST_SPLIT: f"{self.UNICORN_HUGGINGFACE_DATASET_URL}/{self._subject}.json"}

        # Process the test set
        for row in tqdm(
            load_dataset(
                "json",
                data_files=question_data_files,
                split=TEST_SPLIT,
                cache_dir=output_path,
            )
        ):
            # Download the image
            image_path: str = row["image_path"]
            local_image_path: str = os.path.join(output_path, image_path)
            ensure_file_downloaded(
                source_url=self.IMAGE_URL.format(image_path=image_path),
                target_path=local_image_path,
                unpack=False,
            )

            content: List[MediaObject] = [
                MediaObject(location=local_image_path, content_type=f"image/{self._image_type}"),
                MediaObject(text=row["question"], content_type="text/plain"),
            ]
            answer: str = row["answer"]
            instances.append(
                Instance(
                    Input(multimedia_content=MultimediaObject(content)),
                    references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )

        return instances
