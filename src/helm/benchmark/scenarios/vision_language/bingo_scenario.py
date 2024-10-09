import os.path
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
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class BingoScenario(Scenario):
    """
    Holistic Analysis of Hallucination in GPT-4V(ision): Bias and Interference Challenges

    We introduce a new benchmark, namely, the Bias and Interference Challenges in Visual Language Models (Bingo).
    This benchmark is designed to evaluate and shed light on the two common types of hallucinations in visual
    language models: bias and interference. Here, bias refers to the model's tendency to hallucinate certain types
    of responses, possibly due to imbalance in its training data. Interference pertains to scenarios where the
    judgment of GPT-4V(ision) can be disrupted due to how the text prompt is phrased or how the input image is
    presented. The benchmark consists of open-ended question-answer pairs, and we employ open-ended generation
    metrics for evaluation. In the experiment, we identify a notable regional bias, whereby GPT-4V(ision) is
    better at interpreting Western images or images with English writing compared to images from other countries
    or containing text in other languages.


    @article{cui2023holistic,
    title={Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges},
    author={Cui, Chenhang and Zhou, Yiyang and Yang, Xinyu and Wu, Shirley and Zhang, Linjun and
    Zou, James and Yao, Huaxiu},
    journal={arXiv preprint arXiv:2311.03287},
    year={2023}
    }

    Paper: https://arxiv.org/abs/2311.03287
    """

    BINGO_HUGGINGFACE_DATASET_URL: str = "https://huggingface.co/datasets/PahaII/Bingo/resolve/main"

    IMAGE_URL: str = "https://huggingface.co/datasets/PahaII/Bingo/resolve/main/images/{image_path}?download=true"

    SUBJECTS: List[str] = ["T2I", "I2I", "OCR", "Factual", "Region"]

    name = "bingo"
    description = (
        "Evaluate multimodal models on biased and inference-challenging scenarios with five subjects "
        "([Cui et al., 2023](https://arxiv.org/abs/2311.03287))."
    )
    tags = ["vision-language"]

    def __init__(self, subject: str):
        super().__init__()
        assert subject in self.SUBJECTS, f"Invalid subject: {subject}"
        self._subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        images_path: str = os.path.join(output_path, "images")
        ensure_directory_exists(images_path)

        # There is only the test split in Unicorn benchmark
        instances: List[Instance] = []
        question_data_files = {TEST_SPLIT: f"{self.BINGO_HUGGINGFACE_DATASET_URL}/{self._subject}.json"}

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
                MediaObject(location=local_image_path, content_type="image/png"),
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
